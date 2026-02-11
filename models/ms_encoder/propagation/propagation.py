import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.fft import fftshift, ifftshift, fft2, ifft2
# from mnist import MNIST
import random
import matplotlib.pyplot as plt
import cv2

class BaseFrequencySpace(nn.Module):
    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        wavelength_m,
        out_resample_dx_m,
        manual_upsample_factor,
    ):
        """ Initialize a propagation class.
        Args:
            in_size ([L (int), W (int)]): number of input pixels
            in_dx_m ([L (float), W (float)]): input pixel spacing
            out_distance_m (int): propagation distance (z)
            out_size ([L (int), W (int)]): number of output pixels
            out_dx_m ([L (float), W (float)]): output pixel spacing
            wavelength_m (float): wavelength
            out_resample_dx_m (float): resampled output pixel spacing
            manual_upsample_factor (float): input upsampling factor
        """
        super().__init__()
        # Quick checks on the inputs
        out_resample_dx_m = out_dx_m if out_resample_dx_m is None else out_resample_dx_m
        assert manual_upsample_factor >= 1, "manual_upsample factor must >= 1."
        assert isinstance(out_distance_m, float), "out_distance_m must be float."
        assert isinstance(wavelength_m, float), "wavelength_m must be a float."
        for obj in [in_size, in_dx_m, out_size, out_dx_m, out_resample_dx_m]:
            assert len(obj) == 2, "Expected len 2 list for inputs."

        # Move to attributes and convert lists to numpy arrays
        self.in_size = np.array(in_size).astype(int)
        self.in_dx_m = np.array(in_dx_m)
        self.out_distance_m = out_distance_m
        self.out_size = np.array(out_size).astype(int)
        self.out_dx_m = np.array(out_dx_m)
        self.wavelength_m = wavelength_m
        self.out_resample_dx_m = np.array(out_resample_dx_m)
        self.manual_upsample_factor = manual_upsample_factor

        # Run additional assertions required for calculations
        self._validate_inputs()

        # Apply unit conversion from meters to different base
        self.rescale = 1e6  # m to micrometer as our default units
        self._unit_conversion()

    def _validate_inputs(self):
        assert all(
            x <= y for x, y in zip(self.out_dx_m, self.out_resample_dx_m)
        ), "out_resample_dx_m must be geq to out_dx."

    def _unit_conversion(self):
        convert_keys = [
            "in_dx_m",
            "out_distance_m",
            "out_dx_m",
            "out_resample_dx_m",
            "wavelength_m",
        ]
        for key in convert_keys:
            obj = self.__dict__[key]
            new_name = key[:-2]

            if isinstance(obj, float) or isinstance(obj, np.ndarray):
                setattr(self, new_name, obj * self.rescale)
            elif obj is None:
                setattr(self, None)
            else:
                raise ValueError("In key conversion, ran into unknown datatype")

class ASMPropagation(BaseFrequencySpace):
    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        wavelength_m,
        out_resample_dx_m=None,
        manual_upsample_factor=1,
        FFTPadFactor=1.0,
        verbose=False,
    ):
        super().__init__(
            in_size,
            in_dx_m,
            out_distance_m,
            out_size,
            out_dx_m,
            wavelength_m,
            out_resample_dx_m,
            manual_upsample_factor,
        )
        self.verbose = verbose
        self.FFTPadFactor = FFTPadFactor
        self._init_calc_params()

    def _init_calc_params(self, dtype=torch.float32):
        # Padding to odd sized inputs for fourier methods
        # if any(in_size % 2 ==0 for in_size in self.unpadded_in_size):
            # print('Attempting to propagate even number of pixels -> padding to odd for fourier methods.')
        self.unpadded_in_size = self.in_size
        self.unpadded_out_size = self.out_size
        self.out_size = [int(out_size + (in_size+1)%2 * in_dx_m / out_dx_m) 
            for in_size, out_size, in_dx_m, out_dx_m 
            in zip(self.in_size, self.out_size, self.in_dx_m, self.out_dx_m)]
        self.in_size = [insize + (insize+1)%2 for insize in self.in_size]

        # Manual upsampling
        in_dx = self.in_dx
        out_dx = self.out_dx
        calc_in_dx = in_dx / self.manual_upsample_factor
        calc_in_dx = np.where(calc_in_dx < out_dx, calc_in_dx, out_dx)

        in_length = in_dx * self.in_size
        calc_samplesM = np.rint(in_length / calc_in_dx)
        calc_samplesM = np.where(
            np.mod(calc_samplesM, 2) == 0, calc_samplesM + 1, calc_samplesM
        )
        calc_samplesM_r = calc_samplesM[-1] // 2
        self.calc_samplesM = calc_samplesM
        self.calc_samplesM_r = calc_samplesM_r

        self.calc_in_dx = in_length / calc_samplesM
        self.calc_out_dx = self.calc_in_dx

        desired_span = self.out_size * self.out_resample_dx
        current_span = self.calc_out_dx * calc_samplesM
        pad_in = np.rint(
            np.where(
                current_span < desired_span,
                (desired_span - current_span) / 2 / calc_in_dx,
                np.zeros_like(current_span),
            )
        ).astype(int)

        calc_samplesN = 2 * pad_in + calc_samplesM
        calc_samplesN_r = calc_samplesN[-1] // 2 + 1
        self.calc_samplesN = calc_samplesN
        self.calc_samplesN_r = calc_samplesN_r
        self.pad_in = pad_in

        ### AFT Pre-calculation ###
        self.torch_zero = torch.tensor([0.0], dtype=dtype)
        self.padhalf = [int(n * self.FFTPadFactor) for n in self.in_size[-2:]] ## OPTIMIZATION OPTION -- PRECOMPUTE THIS PAD FACTOR
        self.paddings = (self.padhalf[1], self.padhalf[1], self.padhalf[0], self.padhalf[0])

        self.new_gs = [self.calc_samplesN[i] + 2 * self.padhalf[i] for i in range(2)]
        self.x, self.y = cart_grid(self.new_gs, self.calc_in_dx, False)

        self.rarray = torch.sqrt(self.out_distance**2 + self.x**2 + self.y**2)[None, :, :]
        self.angular_wavenumber = 2 * np.pi / torch.tensor(self.wavelength, dtype=dtype)
        self.angular_wavenumber = self.angular_wavenumber[None, None, None]
        self.h = (
            torch.complex(1 / 2 / np.pi * self.out_distance / self.rarray**2, self.torch_zero)
            * torch.complex(1 / self.rarray, -1 * self.angular_wavenumber)
            * torch.exp(torch.complex(self.torch_zero, self.angular_wavenumber * self.rarray))
        )

        self.H = fft2(ifftshift(self.h))
        self.H = torch.exp(torch.complex(self.torch_zero, torch.angle(self.H)))

    def forward(self, amplitude, phase, **kwargs):
        amplitude, phase = self._pad_field(amplitude, phase)

        amplitude, phase = self._regularize_field(amplitude, phase)

        amplitude, phase = self.ASM_transform(amplitude, phase)

        amplitude, phase = self._resample_field(amplitude, phase)

        amplitude, phase = self._unpad_field(amplitude, phase)

        return amplitude, phase

    def ASM_transform(self, amplitude, phase):
        dtype = amplitude.dtype
        device = amplitude.device
        if (self.torch_zero.dtype != dtype):
            self._init_calc_params(dtype=dtype)
        init_shape = amplitude.shape
        self.torch_zero = self.torch_zero.to(device)
        amplitude = F.pad(amplitude.unsqueeze(1), self.paddings, mode="constant", value=0).squeeze(1)
        phase = F.pad(phase.unsqueeze(1), self.paddings, mode="constant", value=0).squeeze(1)

        transform_term = torch.complex(amplitude, self.torch_zero) * torch.exp(
            torch.complex(self.torch_zero, phase)
        )

        angular_spectrum = fft2(ifftshift(transform_term))

        transform_term = angular_spectrum * self.H.to(device)

        outputwavefront = fftshift(ifft2(transform_term))

        outputwavefront = resize_with_crop_or_pad(
            outputwavefront, *init_shape[-2:], False
        )

        return torch.abs(outputwavefront), torch.angle(outputwavefront)

    def _resample_field(self, amplitude, phase):
        scale = tuple(self.calc_out_dx / self.out_resample_dx)

        phase = torch.atan2(
            F.interpolate(torch.sin(phase.unsqueeze(1)), scale_factor=scale, mode="area").squeeze(1),
            F.interpolate(torch.cos(phase.unsqueeze(1)), scale_factor=scale, mode="area").squeeze(1),
        )
        amplitude = F.interpolate(amplitude.unsqueeze(1), scale_factor=scale, mode="area").squeeze(1)

        return amplitude, phase

    def _regularize_field(self, amplitude, phase):
        method = "nearest-exact"
        samplesM = self.calc_samplesM
        pad_in = self.pad_in

        resize_to = samplesM
        resize_to = tuple([int(_) for _ in resize_to])
        amplitude = F.interpolate(amplitude.unsqueeze(1), size=resize_to, mode=method).squeeze(1)
        phase = F.interpolate(phase.unsqueeze(1), size=resize_to, mode=method).squeeze(1)

        paddings = (pad_in[1], pad_in[1], pad_in[0], pad_in[0])
        amplitude = F.pad(amplitude.unsqueeze(1), paddings, mode="constant", value=0).squeeze(1)
        phase = F.pad(phase.unsqueeze(1), paddings, mode="constant", value=0).squeeze(1)

        return amplitude, phase

    def _pad_field(self, amplitude, phase):
        padding = [pad for in_size, amplitude_size in zip(self.in_size, list(amplitude.size())[-2:]) for pad in [0, in_size - amplitude_size] ]
        amplitude = F.pad(amplitude, padding, mode="constant", value=0)
        phase = F.pad(phase, padding, mode="constant", value=0)
        
        assert all(pad == 1 or pad == 0 for pad in padding), f'propagated field must match in_size={self.unpadded_in_size}'

        return amplitude, phase

    def _unpad_field(self, amplitude, phase):
        amplitude = resize_with_crop_or_pad(amplitude, *self.out_size, False)
        phase = resize_with_crop_or_pad(phase, *self.out_size, False)

        amplitude = amplitude[:,:self.unpadded_out_size[0], :self.unpadded_out_size[1]]
        phase = phase[:,:self.unpadded_out_size[0], :self.unpadded_out_size[1]]

        return amplitude, phase

def resize_with_crop_or_pad(input_tensor, target_height, target_width, radial_flag):
    tensor_shape = input_tensor.shape
    input_height = tensor_shape[-2]
    input_width = tensor_shape[-1]

    diffH = target_height - input_height
    diffW = target_width - input_width

    if diffH == 0 and diffW == 0:
        return input_tensor
    
    pad_top = max(diffH // 2, 0)
    pad_bottom = max(diffH - pad_top, 0)
    pad_left = max(diffW // 2, 0)
    pad_right = max(diffW - pad_left, 0)

    padding = [pad_left, pad_right, pad_top, pad_bottom]

    if diffH > 0 or diffW > 0:
        if input_tensor.dim() == 4:
            input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)
        elif input_tensor.dim() == 3:
            padding = [padding[2], padding[3], padding[0], padding[1]]
            input_tensor = F.pad(input_tensor.unsqueeze(0), padding, mode="constant", value=0).squeeze(0)
    
    if diffH < 0 or diffW < 0:
        crop_top = -min(diffH // 2, 0)
        crop_bottom = crop_top + target_height
        crop_left = -min(diffW // 2, 0)
        crop_right = crop_left + target_width

        if input_tensor.dim() == 4:
            input_tensor = input_tensor[:, :, crop_top:crop_bottom, crop_left:crop_right]
        elif input_tensor.dim() == 3:
            input_tensor = input_tensor[:, crop_top:crop_bottom, crop_left:crop_right]

    return input_tensor

def cart_grid(gsize, gdx, radial_symmetry, dtype=torch.float32):
    x, y = torch.meshgrid(
        torch.arange(0, gsize[-1], dtype=dtype),
        torch.arange(0, gsize[-2], dtype=dtype),
        indexing="xy",
    )
    x = x - (x.shape[-1] - 1) / 2
    y = y - (y.shape[-2] - 1) / 2

    x = x * gdx[-1]
    y = y * gdx[-2]
    return x, y

def resize(img, n_pixel):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = len(img)

    if img.shape[-1] != n_pixel:
        out = torch.zeros(batch_size, n_pixel, n_pixel)
        for i, im in enumerate(img):
            out[i] = torch.from_numpy(cv2.resize(im, (n_pixel, n_pixel)))
        img = out.to(device)

    return img

# Replace the test_ASMPropagation function in propagation.py with this version:

# def test_ASMPropagation():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     mnist = MNIST('./data/MNIST/raw')
#     images, labels = mnist.load_testing()
#     images = np.array(images) / 255.0
#     images = images.reshape(-1, 28, 28)
#     num_imgs = 4
    
#     random_indices = random.sample(range(len(images)), num_imgs)
#     test_images = np.array([images[i] for i in random_indices])

#     invert_img_intensity = False
#     in_size = 200.0
#     n_pixel_input = 200
#     input_dx = in_size / n_pixel_input
#     out_size = 200.0
#     n_pixel_output = 200
#     output_dx = out_size / n_pixel_output

#     # Resize the handwritten digits
#     x_amp = resize(test_images, n_pixel_input)
#     x_phase = torch.zeros_like(x_amp).to(device)

#     # Create a square aperture
#     aperture = np.zeros((n_pixel_input, n_pixel_input), dtype=np.float32)
#     square_size = int(100.0 / input_dx)  # Convert 100x100 um to pixels
#     center = n_pixel_input // 2
#     half_size = square_size // 2

#     aperture[
#         center - half_size:center + half_size,
#         center - half_size:center + half_size
#     ] = 1.0

#     aperture = torch.tensor(aperture, device=device).unsqueeze(0)  # Add batch dimension

#     # Propagation parameters
#     in_size = [n_pixel_input, n_pixel_input]
#     in_dx_m = [input_dx, input_dx]
#     out_distance_m = 200.0
#     out_size = [n_pixel_output, n_pixel_output]
#     out_dx_m = [output_dx, output_dx]
#     wavelength_m = 1.0

#     propagator = ASMPropagation(
#         in_size,
#         in_dx_m,
#         out_distance_m,
#         out_size,
#         out_dx_m,
#         wavelength_m,
#     )

#     print("\n" + "="*60)
#     print("ASM PROPAGATION ENERGY ANALYSIS")
#     print("="*60)
    
#     # Track energy for digits
#     print("\nDigit Propagation:")
#     for i in range(num_imgs):
#         input_energy = (x_amp[i] ** 2).sum().item()
        
#         # Propagate single image
#         amp_in = x_amp[i:i+1]
#         phase_in = x_phase[i:i+1]
#         amplitude_out, phase_out = propagator(amp_in, phase_in)
        
#         output_energy = (amplitude_out ** 2).sum().item()
#         efficiency = output_energy / (input_energy + 1e-10)
        
#         print(f"  Image {i+1} (Label: {labels[random_indices[i]]}):")
#         print(f"    Input energy:  {input_energy:.4f}")
#         print(f"    Output energy: {output_energy:.4f}")
#         print(f"    Efficiency:    {efficiency:.4f}")
#         if efficiency > 1.1:
#             print(f"    ⚠️  WARNING: Energy amplification detected!")
#         elif efficiency < 0.001:
#             print(f"    ⚠️  WARNING: Severe energy loss detected!")
    
#     # Track energy for square aperture
#     print("\nSquare Aperture Propagation:")
#     input_energy_aperture = (aperture ** 2).sum().item()
#     amplitude_aperture, phase_aperture = propagator(aperture, torch.zeros_like(aperture).to(device))
#     output_energy_aperture = (amplitude_aperture ** 2).sum().item()
#     efficiency_aperture = output_energy_aperture / (input_energy_aperture + 1e-10)
    
#     print(f"  Input energy:  {input_energy_aperture:.4f}")
#     print(f"  Output energy: {output_energy_aperture:.4f}")
#     print(f"  Efficiency:    {efficiency_aperture:.4f}")
#     if efficiency_aperture > 1.1:
#         print(f"  ⚠️  WARNING: Energy amplification detected!")
#     elif efficiency_aperture < 0.001:
#         print(f"  ⚠️  WARNING: Severe energy loss detected!")
    
#     # Debug the transfer function
#     print("\nTransfer Function Analysis:")
#     if hasattr(propagator, 'H'):
#         H_magnitude = torch.abs(propagator.H)
#         print(f"  |H| max: {H_magnitude.max().item():.6f}")
#         print(f"  |H| min: {H_magnitude.min().item():.6f}")
#         print(f"  |H| mean: {H_magnitude.mean().item():.6f}")
#         print(f"  % of |H| > 1: {((H_magnitude > 1).sum() / H_magnitude.numel() * 100):.2f}%")
    
#     print("="*60 + "\n")

#     # Propagate all digits again for plotting (since we did them one by one above)
#     amplitude_digits, phase_digits = propagator(x_amp, x_phase)
    
#     # Convert to numpy for plotting
#     amplitude_digits = amplitude_digits.cpu().detach().numpy()
#     phase_digits = phase_digits.cpu().detach().numpy()
#     amplitude_aperture = amplitude_aperture.cpu().detach().numpy()
#     phase_aperture = phase_aperture.cpu().detach().numpy()

#     # Original plotting code continues here...
#     fig, axs = plt.subplots(num_imgs + 1, 3, figsize=(15, 12))
    
#     for i in range(num_imgs):
#         axs[i, 0].imshow(x_amp.cpu().detach().numpy()[i], cmap='gray')
#         axs[i, 0].set_title(f'Original Label: {labels[random_indices[i]]}')
#         axs[i, 0].axis('off')

#         axs[i, 1].imshow(amplitude_digits[i], cmap='jet')
#         axs[i, 1].set_title('Propagated Amplitude')
#         axs[i, 1].axis('off')

#         axs[i, 2].imshow(phase_digits[i], cmap='jet')
#         axs[i, 2].set_title('Propagated Phase')
#         axs[i, 2].axis('off')

#     # Plot the square aperture results
#     axs[num_imgs, 0].imshow(aperture.cpu().detach().numpy()[0], cmap='gray')
#     axs[num_imgs, 0].set_title('Square Aperture')
#     axs[num_imgs, 0].axis('off')

#     axs[num_imgs, 1].imshow(amplitude_aperture[0], cmap='jet')
#     axs[num_imgs, 1].set_title('Propagated Amplitude')
#     axs[num_imgs, 1].axis('off')

#     axs[num_imgs, 2].imshow(phase_aperture[0], cmap='jet')
#     axs[num_imgs, 2].set_title('Propagated Phase')
#     axs[num_imgs, 2].axis('off')

#     plt.tight_layout()
#     plt.show()


# # Also add this simpler test function for quick debugging:
# def test_propagation_energy_only():
#     """Quick test to check energy conservation in propagation."""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Simple test parameters
#     propagator = ASMPropagation(
#         in_size=[100, 100],
#         in_dx_m=[1.0, 1.0],
#         out_distance_m=50.0,
#         out_size=[100, 100],
#         out_dx_m=[1.0, 1.0],
#         wavelength_m=0.5,
#     )
    
#     # Create simple test patterns
#     test_patterns = {
#         'Gaussian': None,
#         'Plane Wave': None,
#         'Point Source': None,
#     }
    
#     x = torch.linspace(-50, 50, 100)
#     X, Y = torch.meshgrid(x, x, indexing='ij')
    
#     # Gaussian
#     test_patterns['Gaussian'] = torch.exp(-(X**2 + Y**2) / 200).unsqueeze(0).to(device)
    
#     # Plane wave (uniform amplitude)
#     test_patterns['Plane Wave'] = torch.ones(1, 100, 100).to(device)
    
#     # Point source
#     point = torch.zeros(1, 100, 100).to(device)
#     point[0, 50, 50] = 10.0
#     test_patterns['Point Source'] = point
    
#     print("\n" + "="*40)
#     print("QUICK PROPAGATION ENERGY TEST")
#     print("="*40)
    
#     for name, pattern in test_patterns.items():
#         input_energy = (pattern ** 2).sum().item()
#         output_amp, _ = propagator(pattern, torch.zeros_like(pattern))
#         output_energy = (output_amp ** 2).sum().item()
#         efficiency = output_energy / (input_energy + 1e-10)
        
#         print(f"\n{name}:")
#         print(f"  Input energy:  {input_energy:.4f}")
#         print(f"  Output energy: {output_energy:.4f}")
#         print(f"  Efficiency:    {efficiency:.4f}")
        
#         if efficiency > 1.1:
#             print(f"  ❌ FAIL: Energy amplification!")
#         elif efficiency < 0.001:
#             print(f"  ❌ FAIL: Energy vanished!")
#         else:
#             print(f"  ✅ PASS: Energy conserved")
    
#     print("="*40)


# if __name__ == "__main__":
#     test_ASMPropagation()
