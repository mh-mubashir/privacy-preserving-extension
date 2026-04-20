"""
VQ-VAE ARL wrapper (integrates Member 3 Malia's implementation).

Changes from original:
  - VectorQuantizer now uses EMA updates to prevent codebook collapse
  - embedding_loss is stored and added to enc_loss in adversarial_training.py
    via the vae_weight * recon_l term (recon_l now includes codebook loss)
  - sigmoid added to decoder output to keep values in [0, 1]

forward(x)                  -> recon in [0, 1]
forward(x, return_aux=True) -> (recon, None, None, z_q)
    mu and logvar are None (VQ-VAE has no Gaussian posterior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, 3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, 1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.res_block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)


class VQEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, 3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def forward(self, x):
        return self.net(x)


class VQDecoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, 3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 3, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    """
    EMA-based VQ codebook — prevents codebook collapse.

    Instead of updating embeddings via gradients (which causes collapse),
    EMA directly tracks encoder output statistics per codebook entry.
    Only the commitment loss is backpropagated through the encoder.

    Embeddings are initialized lazily from the first batch of encoder
    outputs rather than random noise — this prevents early collapse where
    all inputs map to the same few embeddings before training begins.
    """
    def __init__(self, n_e, e_dim, beta, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.initialized = False

        embedding = torch.randn(n_e, e_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)

        # Lazy initialization: set embeddings from first real encoder outputs
        # so the codebook starts near the actual data distribution
        if not self.initialized and self.training:
            n = z_flat.shape[0]
            if n >= self.n_e:
                indices = torch.randperm(n)[:self.n_e]
                self.embedding.data.copy_(z_flat[indices].detach())
            else:
                # fewer samples than embeddings — tile and add small noise
                repeats = (self.n_e + n - 1) // n
                tiled = z_flat.detach().repeat(repeats, 1)[:self.n_e]
                self.embedding.data.copy_(tiled + 0.01 * torch.randn_like(tiled))
            self.embedding_avg.data.copy_(self.embedding.data)
            self.initialized = True

        d = (z_flat.pow(2).sum(1, keepdim=True)
             + self.embedding.pow(2).sum(1)
             - 2 * z_flat @ self.embedding.t())

        idx = torch.argmin(d, dim=1).unsqueeze(1)
        one_hot = torch.zeros(idx.shape[0], self.n_e, device=z.device)
        one_hot.scatter_(1, idx, 1)

        z_q = (one_hot @ self.embedding).view(z.shape)

        # EMA update (only during training)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                one_hot.sum(0), alpha=1 - self.decay)
            dw = one_hot.t() @ z_flat
            self.embedding_avg.data.mul_(self.decay).add_(
                dw, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.n_e * self.eps) * n
            )
            self.embedding.data.copy_(
                self.embedding_avg / cluster_size.unsqueeze(1))

        # commitment loss only — codebook updated via EMA
        loss = self.beta * (z_q.detach() - z).pow(2).mean()

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        e_mean = one_hot.mean(0)
        perplexity = torch.exp(-(e_mean * (e_mean + 1e-10).log()).sum())

        return loss, z_q.permute(0, 3, 1, 2).contiguous(), perplexity


class VQVAEWrapper(nn.Module):
    """
    ARL-compatible VQ-VAE with EMA codebook updates.

    forward(x)                  -> recon in [0, 1]
    forward(x, return_aux=True) -> (recon, None, None, z_q)
        mu/logvar are None — VQ-VAE has no Gaussian posterior.
        embedding_loss is stored in self.last_codebook_loss for use
        in adversarial_training.py's vae_weight term.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        h_dim=128,
        res_h_dim=32,
        n_res_layers=2,
        n_embeddings=256,
        embedding_dim=64,
        beta=0.5,
        img_size=224,
    ):
        super().__init__()
        self.encoder       = VQEncoder(in_channels, h_dim, n_res_layers, res_h_dim)
        self.pre_quant     = nn.Conv2d(h_dim, embedding_dim, 1)
        self.vq            = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder       = VQDecoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.embedding_dim = embedding_dim
        self.last_codebook_loss = None  # stored for adversarial_training.py

    def forward(self, x, return_aux=False):
        z_e = self.pre_quant(self.encoder(x))
        codebook_loss, z_q, perplexity = self.vq(z_e)
        self.last_codebook_loss = codebook_loss  # save for training loop
        recon = torch.sigmoid(self.decoder(z_q))
        if return_aux:
            return recon, None, None, z_q
        return recon

    def get_latent(self, x):
        z_e = self.pre_quant(self.encoder(x))
        _, z_q, _ = self.vq(z_e)
        return z_q