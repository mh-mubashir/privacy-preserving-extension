import torch
from diffusers import DiffusionPipeline

class myDiffusionPipeline(DiffusionPipeline):
    def __init__(self, unet, vae, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: torch.Tensor,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ):
        """
        简化后的 i2i 逻辑：先根据 strength 确定要从多少步开始反向扩散，
        对初始 latent 添加噪声，再进行反向扩散。
        """
        device = next(self.unet.parameters()).device
        batch_size = image.shape[0]

        # 1. 根据 prompt 生成条件文本 embedding
        input_ids = self.tokenizer(
            prompt,  # 若 prompt 是单字符串，需要重复 match batch_size
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        text_embeds = self.text_encoder(input_ids)[0]

        # 2. 将输入图像编码到 latent 空间
        #    VAE 输出的分布先进行采样，再乘以 scaling_factor
        init_latent = self.vae.encode(image.to(device)).latent_dist.sample() * self.vae.config.scaling_factor

        # 3. 设置扩散步数（scheduler 通常在 pipeline.__init__ 或外部已 set_format；确保不冲突）
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 4. 计算起始步数（与 strength 对应），并截取 timesteps
        #    注意：如果 scheduler.timesteps 是从大到小，需要取后面一段；反之亦然
        init_timestep = int(num_inference_steps * strength)
        init_timestep = min(init_timestep, num_inference_steps)  # 防止越界
        timesteps = self.scheduler.timesteps[-init_timestep:]    # 取后面 init_timestep 个时间步

        # 5. 对初始 latent 添加噪声，使其对应到 init_timestep
        noise = torch.randn_like(init_latent)
        latents = self.scheduler.add_noise(init_latent, noise, timesteps[0])

        # 6. 如果使用 classifier-free guidance，则生成无条件 embedding 并拼接
        if guidance_scale > 1.0:
            uncond_input = self.tokenizer(
                ["" for _ in range(batch_size)],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            uncond_embeds = self.text_encoder(uncond_input)[0]
            text_embeds = torch.cat([uncond_embeds, text_embeds], dim=0)

        # 7. 扩散去噪迭代过程（从 init_timestep 到 0）
        for i, t in enumerate(timesteps):
            # (7.1) 如果使用 guidance，需要将 latents 在 batch 维度上复制一份
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents, latents], dim=0)
            else:
                latent_model_input = latents

            # (7.2) 根据 scheduler 调整输入
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # (7.3) 使用 UNet 预测噪声
            with torch.no_grad():    
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)[0]

            # (7.4) guidance 结合
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # (7.5) 根据噪声更新 latent
            step_result = self.scheduler.step(noise_pred, t, latents)
            latents = step_result[0]

        # 8. 解码生成图片
        latents = latents / self.vae.config.scaling_factor
        decoded = self.vae.decode(latents)[0]

        # 9. 将范围从 [-1,1] 映射到 [0,1]，并返回
        image = (decoded / 2 + 0.5).clamp(0, 1)

        return image 

