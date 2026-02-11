import numpy as np
import torch
import torch.nn as nn
from  import propagation as lp
from models.ms_encoder.utils import plots


class Encoder(nn.Module):
    def __init__(self, 
                 wavelength=0.7,
                 n_pixel_input=28,
                 n_pixel_MS=200, 
                 n_pixel_output=200,
                 size_x_input=100,
                 size_x_MS=100,
                 size_x_output=100,
                 dz_input_MS=40,
                 dz_MS_output=40,
                 random_seed=42,
                 savedir=None):
        """
        Optical Encoder with fixed random phase metasurface.
        
        Args:
            wavelength: Wavelength in micrometers
            n_pixel_input: Number of pixels for input image (1D)
            n_pixel_MS: Number of pixels in metasurface (1D)
            n_pixel_output: Number of pixels in output plane (1D)
            size_x_input: Lateral size of input image in micrometers
            size_x_MS: Lateral size of metasurface in micrometers
            size_x_output: Lateral size of output plane in micrometers
            dz_input_MS: Distance from input to metasurface in micrometers
            dz_MS_output: Distance from metasurface to output in micrometers
            random_seed: Random seed for phase generation
            savedir: Directory to save plots (optional)
        """
        super(Encoder, self).__init__()
        
        # Store parameters
        self.wavelength = wavelength
        self.n_pixel_input = n_pixel_input
        self.n_pixel_MS = n_pixel_MS
        self.n_pixel_output = n_pixel_output
        self.size_x_input = size_x_input
        self.size_x_MS = size_x_MS
        self.size_x_output = size_x_output
        self.dz_input_MS = dz_input_MS
        self.dz_MS_output = dz_MS_output
        self.random_seed = random_seed
        self.savedir = savedir
        
        # Calculate pixel sizes
        self.dx_input = size_x_input / n_pixel_input
        self.dx_MS = size_x_MS / n_pixel_MS
        self.dx_output = size_x_output / n_pixel_output
        
        # Initialize metasurface with fixed amplitude and random phase
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Fixed amplitude = 1
        self.ms_amplitude = nn.Parameter(
            torch.ones(n_pixel_MS, n_pixel_MS), 
            requires_grad=False
        )
        
        # Random phase [0, 2π]
        self.ms_phase = nn.Parameter(
            2 * np.pi * torch.rand(n_pixel_MS, n_pixel_MS),
            requires_grad=False
        )
        
        # Initialize propagation modules
        self.asm_input_MS = lp.ASMPropagation(
            in_size=[self.n_pixel_input, self.n_pixel_input],
            in_dx_m=[self.dx_input, self.dx_input],
            out_distance_m=self.dz_input_MS,
            out_size=[self.n_pixel_MS, self.n_pixel_MS],
            out_dx_m=[self.dx_MS, self.dx_MS],
            wavelength_m=self.wavelength,
            verbose=False
        )
        
        self.asm_MS_output = lp.ASMPropagation(
            in_size=[self.n_pixel_MS, self.n_pixel_MS],
            in_dx_m=[self.dx_MS, self.dx_MS],
            out_distance_m=self.dz_MS_output,
            out_size=[self.n_pixel_output, self.n_pixel_output],
            out_dx_m=[self.dx_output, self.dx_output],
            wavelength_m=self.wavelength,
            verbose=False
        )
    
    def forward(self, x, plot=False, idx=0):
        """
        Forward propagation through encoder.
        
        Args:
            x: Input images [batch_size, 2, n_pixel_input, n_pixel_input]
               where channel 0 is amplitude and channel 1 is phase
            plot: Whether to plot intermediate results
            idx: Index in batch to plot
            
        Returns:
            encoded_amplitude: Output amplitude [batch_size, n_pixel_output, n_pixel_output]
            encoded_phase: Output phase [batch_size, n_pixel_output, n_pixel_output]
        """
        device = x.device
        
        # Extract amplitude and phase from input
        x_amp = x[:, 0, :, :].to(device)
        x_phase = x[:, 1, :, :].to(device)
        
        if plot and self.savedir:
            # Plot input
            area_per_pixel = (self.size_x_input / self.n_pixel_input) ** 2
            plots.input_layer(area_per_pixel * (x_amp[idx]**2).cpu(), self.savedir)
        
        # Propagate from input to metasurface
        x_amp, x_phase = self.asm_input_MS(x_amp, x_phase)
        
        # Apply metasurface modulation
        ms_amp = self.ms_amplitude.to(device)
        ms_phase = self.ms_phase.to(device)
        
        if plot and self.savedir:
            # Store for plotting
            wavefront_before_amp = x_amp[idx].cpu()
            wavefront_before_phase = x_phase[idx].cpu()
            ms_amp_plot = ms_amp.cpu()
            ms_phase_plot = ms_phase.cpu()
        
        # Modulate wavefront with metasurface
        x_amp = ms_amp * x_amp
        x_phase = ms_phase + x_phase
        
        # Propagate from metasurface to output
        encoded_amp, encoded_phase = self.asm_MS_output(x_amp, x_phase)
        
        if plot and self.savedir:
            # Plot metasurface and wavefront
            self._plot_encoder_layers(
                wavefront_before_amp, wavefront_before_phase,
                ms_amp_plot, ms_phase_plot,
                encoded_amp[idx].cpu(), encoded_phase[idx].cpu()
            )
        
        return encoded_amp, encoded_phase
    
    def _plot_encoder_layers(self, wf_amp, wf_phase, ms_amp, ms_phase, 
                            out_amp, out_phase):
        """Plot encoder layers for visualization."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 15), dpi=150)
        
        # Custom colormap for amplitude
        colors = [(0, 0, 0), (1, 0, 1)]
        cmap_amp = LinearSegmentedColormap.from_list('custom', colors, N=100)
        
        # Row 1: Wavefront before metasurface
        im1 = axes[0, 0].imshow(wf_amp, cmap=cmap_amp, vmin=0, vmax=1)
        axes[0, 0].set_title('Wavefront Amplitude (before MS)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        im2 = axes[0, 1].imshow(wf_phase, cmap='twilight', vmin=0, vmax=2*np.pi)
        axes[0, 1].set_title('Wavefront Phase (before MS)')
        axes[0, 1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        cbar2.set_ticks([0, np.pi, 2*np.pi])
        cbar2.set_ticklabels(['0', 'π', '2π'])
        
        # Row 2: Metasurface
        im3 = axes[1, 0].imshow(ms_amp, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Metasurface Amplitude (fixed = 1)')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        im4 = axes[1, 1].imshow(ms_phase, cmap='twilight', vmin=0, vmax=2*np.pi)
        axes[1, 1].set_title(f'Metasurface Phase (seed={self.random_seed})')
        axes[1, 1].axis('off')
        cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        cbar4.set_ticks([0, np.pi, 2*np.pi])
        cbar4.set_ticklabels(['0', 'π', '2π'])
        
        # Row 3: Output
        im5 = axes[2, 0].imshow(out_amp, cmap=cmap_amp)
        axes[2, 0].set_title('Encoded Amplitude (output)')
        axes[2, 0].axis('off')
        plt.colorbar(im5, ax=axes[2, 0], fraction=0.046)
        
        im6 = axes[2, 1].imshow(out_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[2, 1].set_title('Encoded Phase (output)')
        axes[2, 1].axis('off')
        cbar6 = plt.colorbar(im6, ax=axes[2, 1], fraction=0.046)
        cbar6.set_ticks([-np.pi, 0, np.pi])
        cbar6.set_ticklabels(['-π', '0', 'π'])
        
        plt.suptitle('Encoder Propagation Visualization', fontsize=16)
        plt.tight_layout()
        
        if self.savedir:
            plt.savefig(f'{self.savedir}/encoder_layers.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_encoding_size(self):
        """Return the size of the encoded output."""
        return self.n_pixel_output
    
    def encode_batch(self, images):
        """
        Convenience method to encode a batch of images and return complex field.
        
        Args:
            images: Input batch [batch_size, 2, H, W]
            
        Returns:
            complex_field: Complex encoded field [batch_size, n_pixel_output, n_pixel_output]
        """
        amp, phase = self.forward(images)
        return amp * torch.exp(1j * phase)


# Example usage and test function
if __name__ == "__main__":
    # Test the encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create encoder instance
    encoder = Encoder(
        wavelength=0.7,
        n_pixel_input=28,
        n_pixel_MS=100,
        n_pixel_output=100,
        size_x_input=100,
        size_x_MS=100,
        size_x_output=100,
        dz_input_MS=40,
        dz_MS_output=1,
        random_seed=42,
        savedir='./encoder_test'
    )
    
    encoder = encoder.to(device)
    
    # Create test input (batch of 4 images)
    batch_size = 4
    test_input = torch.zeros(batch_size, 2, 28, 28).to(device)
    
    # Create simple test patterns
    for i in range(batch_size):
        # Create different patterns for each image
        if i == 0:  # Centered square
            test_input[i, 0, 10:18, 10:18] = 1.0
        elif i == 1:  # Vertical line
            test_input[i, 0, :, 13:15] = 1.0
        elif i == 2:  # Horizontal line
            test_input[i, 0, 13:15, :] = 1.0
        elif i == 3:  # Diagonal
            for j in range(28):
                if 0 <= j < 28:
                    test_input[i, 0, j, j] = 1.0
        
        # Phase is initially zero
        test_input[i, 1, :, :] = 0.0
    
    # Run encoding
    print("Testing encoder...")
    encoded_amp, encoded_phase = encoder(test_input, plot=True, idx=0)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Encoded amplitude shape: {encoded_amp.shape}")
    print(f"Encoded phase shape: {encoded_phase.shape}")
    
    # Calculate some statistics
    for i in range(batch_size):
        input_energy = (test_input[i, 0]**2).sum().item()
        output_energy = (encoded_amp[i]**2).sum().item()
        efficiency = output_energy / (input_energy + 1e-10)
        print(f"\nImage {i}:")
        print(f"  Input energy: {input_energy:.4f}")
        print(f"  Output energy: {output_energy:.4f}")
        print(f"  Efficiency: {efficiency:.4f}")