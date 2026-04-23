import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# code modified from https://github.com/MishaLaskin/vqvae/tree/master

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings=256, embedding_dim=64, beta=0.5,
                 save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False, return_aux=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)

        # return_aux=True used by adversarial_training.py:
        # recon, _, _, z_q = encoder_model(inputs, return_aux=True)
        if return_aux:
            return x_hat, embedding_loss, perplexity, z_q

        return embedding_loss, x_hat, perplexity


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        # 2 downsampling layers: 64 -> 32 -> 16
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)


class VectorQuantizer(nn.Module):
    """
    VQ-VAE codebook with EMA (Exponential Moving Average) updates.
    EMA updates prevent codebook collapse by directly updating embeddings
    rather than relying on gradients, which stabilizes training significantly.
    """
    def __init__(self, n_e, e_dim, beta, decay=0.99, eps=1e-5):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # Codebook embeddings
        embedding = torch.randn(n_e, e_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding).view(z.shape)

        # EMA codebook update (only during training)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                min_encodings.sum(0), alpha=1 - self.decay)
            dw = torch.matmul(min_encodings.t(), z_flattened)
            self.embedding_avg.data.mul_(self.decay).add_(
                dw, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.n_e * self.eps) * n
            )
            self.embedding.data.copy_(
                self.embedding_avg / cluster_size.unsqueeze(1))

        # commitment loss only (codebook updated via EMA, not gradient)
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients with straight-through estimator
        z_q = z + (z_q - z).detach()

        # perplexity — measures codebook usage (want this HIGH, close to n_e)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        # Mirror of encoder: upsample 16 -> 32 -> 64
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    import utilsVqvae as utils

    parser = argparse.ArgumentParser()
    timestamp = utils.readable_timestamp()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=5000)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset", type=str, default='CIFAR10')
    parser.add_argument("-save", action="store_true")
    parser.add_argument("--filename", type=str, default=timestamp)

    args = parser.parse_args()

    if args.save:
        print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

    training_data, validation_data, training_loader, validation_loader, x_train_var = \
        utils.load_data_and_data_loaders(args.dataset, args.batch_size)

    model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
                  args.n_residual_layers, args.n_embeddings,
                  args.embedding_dim, args.beta).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    model.train()

    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }

    def validate():
        model.eval()
        val_recon_errors, val_loss_vals = [], []
        with torch.no_grad():
            for (x, _) in validation_loader:
                x = x.to(device)
                embedding_loss, x_hat, _ = model(x)
                recon_loss = torch.mean((x_hat - x)**2) / x_train_var
                loss = recon_loss + embedding_loss
                val_recon_errors.append(recon_loss.cpu().numpy())
                val_loss_vals.append(loss.cpu().numpy())
        model.train()
        return np.mean(val_recon_errors), np.mean(val_loss_vals)

    def train():
        loader_iter = iter(training_loader)
        for i in range(args.n_updates):
            try:
                (x, _) = next(loader_iter)
            except StopIteration:
                loader_iter = iter(training_loader)
                (x, _) = next(loader_iter)

            x = x.to(device)
            optimizer.zero_grad()

            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = i

            if i % args.log_interval == 0:
                if args.save:
                    utils.save_model_and_results(model, results, args.__dict__, args.filename)
                val_recon, val_loss = validate()
                print('Update #', i,
                      '| Train Recon Error:', np.mean(results["recon_errors"][-args.log_interval:]),
                      '| Train Loss:', np.mean(results["loss_vals"][-args.log_interval:]),
                      '| Perplexity:', np.mean(results["perplexities"][-args.log_interval:]),
                      '| Val Recon Error:', val_recon,
                      '| Val Loss:', val_loss)

    train()