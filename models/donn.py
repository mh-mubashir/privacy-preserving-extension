import lightridge
import lightridge.layers as layers
import lightridge.utils as utils
import lightridge.data as dataset
from   lightridge.get_h import _field_Fresnel
import torch.nn as nn

from time import time
import torch
import torch.nn.functional as F

# Parameters define
batch_size = 250
sys_size = 192
distance = 0.3
pixel_size = 3.6e-5
pad = 100
wavelength = 5.32e-7
approx = 'Fresnel'
amp_factor = 1.5
depth = 5
device = "cuda:0"
epochs = 10
lr = 0.1
det_x_loc = [40, 40, 40, 90, 90, 90, 90, 140, 140, 140]
det_y_loc = [40, 90, 140, 30, 70, 110, 150, 40, 90, 140]
det_size = 20

class DiffractiveClassifier_Raw(torch.nn.Module):
    def __init__(self, use_det=False, det_x_loc=None, det_y_loc=None, det_size=None, 
                 wavelength=5.32e-7, pixel_size=0.000036,
                 batch_norm=False, sys_size = 200, 
                 pad = 100, distance=0.1, num_layers=2, amp_factor=6, approx="Fresnel3"):
        super(DiffractiveClassifier_Raw, self).__init__()
        self.amp_factor = amp_factor
        self.size = sys_size
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pad = pad
        self.approx=approx
        self.use_det = use_det
        
        if batch_norm == True:
            raise NotImplementedError("Batch normalization is not implemented yet.")
        
        self.diffractive_layers = torch.nn.ModuleList([layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                                                pixel_size=self.pixel_size,
                                                                                size=self.size, 
                                                                                pad = self.pad, 
                                                                                distance=self.distance,
                                                                                amplitude_factor = amp_factor, 
                                                                                approx=self.approx,
                                                                                phase_mod=True) for _ in range(num_layers)])
        
        self.last_diffraction = layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                         pixel_size=self.pixel_size,
                                                         size=self.size, 
                                                         pad = self.pad, 
                                                         distance=self.distance,
                                                         approx=self.approx, 
                                                         phase_mod=False)
        
        if self.use_det:
            self.detector = layers.Detector(x_loc=det_x_loc, y_loc=det_y_loc, det_size=det_size, size=self.size)

    def forward(self, x):
        for index, layer in enumerate(self.diffractive_layers):
            x = layer(x)
        x = self.last_diffraction(x)
        
        if self.use_det:
            output = self.detector(x)
            return output
        else:
            return F.hardtanh(x.abs(), 0, 1)

    def prop_view(self, x):
        prop_list = []
        prop_list.append(x)
        x = x #* self.amp_factor
        for index, layer in enumerate(self.diffractive_layers):
            x = layer(x)
            prop_list.append(x)
        x = self.last_diffraction(x)
        prop_list.append(x)
        for i in range(x.shape[0]):
            print(i)
            utils.forward_func_visualization(prop_list, self.size, fname="mnist_%s.pdf" % i, idx=i, intensity_plot=False)
        
        if self.use_det:
            output = self.detector(x)
        return

    def phase_view(self, cmap="hsv"):
        phase_list = []
        for index, layer in enumerate(self.diffractive_layers):
            phase_list.append(layer.phase)
        print(phase_list[0].shape)
        utils.phase_visualization(phase_list,size=self.size, cmap=cmap, fname="prop_view_reflection.pdf")
        return
    

class DiffractiveClassifier_RGB(torch.nn.Module):
    def __init__(self, use_det=False, det_x_loc=None, det_y_loc=None, det_size=None, 
                 wavelength=5.32e-7, pixel_size=0.000036,
                 batch_norm=False, sys_size = 200, 
                 pad = 100, distance=0.1, num_layers=2, amp_factor=6, approx="Fresnel3"):
        super(DiffractiveClassifier_RGB, self).__init__()
        self.amp_factor = amp_factor
        self.size = sys_size
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pad = pad
        self.approx=approx
        self.use_det = use_det
        
        if batch_norm == True:
            raise NotImplementedError("Batch normalization is not implemented yet.")
        
        
        self.diffractive_layers_r = torch.nn.ModuleList([layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                                                pixel_size=self.pixel_size,
                                                                                size=self.size, 
                                                                                pad = self.pad, 
                                                                                distance=self.distance,
                                                                                amplitude_factor = amp_factor, 
                                                                                approx=self.approx,
                                                                                phase_mod=True) for _ in range(num_layers)])
        self.diffractive_layers_g = torch.nn.ModuleList([layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                                                pixel_size=self.pixel_size,
                                                                                size=self.size, 
                                                                                pad = self.pad, 
                                                                                distance=self.distance,
                                                                                amplitude_factor = amp_factor, 
                                                                                approx=self.approx,
                                                                                phase_mod=True) for _ in range(num_layers)])
        self.diffractive_layers_b = torch.nn.ModuleList([layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                                                pixel_size=self.pixel_size,
                                                                                size=self.size, 
                                                                                pad = self.pad, 
                                                                                distance=self.distance,
                                                                                amplitude_factor = amp_factor, 
                                                                                approx=self.approx,
                                                                                phase_mod=True) for _ in range(num_layers)])
        
        self.last_diffraction_r = layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                         pixel_size=self.pixel_size,
                                                         size=self.size, 
                                                         pad = self.pad, 
                                                         distance=self.distance,
                                                         approx=self.approx, 
                                                         phase_mod=False)
        self.last_diffraction_g = layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                         pixel_size=self.pixel_size,
                                                         size=self.size, 
                                                         pad = self.pad, 
                                                         distance=self.distance,
                                                         approx=self.approx, 
                                                         phase_mod=False)
        self.last_diffraction_b = layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                         pixel_size=self.pixel_size,
                                                         size=self.size, 
                                                         pad = self.pad, 
                                                         distance=self.distance,
                                                         approx=self.approx, 
                                                         phase_mod=False)
        
        if self.use_det:
            self.detector = layers.Detector(x_loc=det_x_loc, y_loc=det_y_loc, det_size=det_size, size=self.size)

    def forward(self, x, no_clamp=False):
        xr, xg, xb = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        
        for index, layer in enumerate(self.diffractive_layers_r):
            xr = layer(xr)
        xr = self.last_diffraction_r(xr)
        
        for index, layer in enumerate(self.diffractive_layers_g):
            xg = layer(xg)
        xg = self.last_diffraction_g(xg)
        
        for index, layer in enumerate(self.diffractive_layers_b):
            xb = layer(xb)
        xb = self.last_diffraction_b(xb)
        
        x = torch.stack([xr,xg,xb], dim=1)[:, :, 0]
        
        if self.use_det:
            output = self.detector(x)
            return output
        else:
            if no_clamp:
                return x.abs()
            return F.hardtanh(x.abs(), 0, 1)

    def prop_view(self, x):
        prop_list = []
        prop_list.append(x)
        x = x #* self.amp_factor
        for index, layer in enumerate(self.diffractive_layers):
            x = layer(x)
            prop_list.append(x)
        x = self.last_diffraction(x)
        prop_list.append(x)
        for i in range(x.shape[0]):
            print(i)
            utils.forward_func_visualization(prop_list, self.size, fname="mnist_%s.pdf" % i, idx=i, intensity_plot=False)
        
        if self.use_det:
            output = self.detector(x)
        return

    def phase_view(self, cmap="hsv"):
        phase_list = []
        for index, layer in enumerate(self.diffractive_layers):
            phase_list.append(layer.phase)
        print(phase_list[0].shape)
        utils.phase_visualization(phase_list,size=self.size, cmap=cmap, fname="prop_view_reflection.pdf")
        return
    
class DiffractiveClassifier_RGB_residual(torch.nn.Module):
    def __init__(self, use_det=False, det_x_loc=None, det_y_loc=None, det_size=None, 
                 wavelength=5.32e-7, pixel_size=0.000036,
                 batch_norm=False, sys_size = 200, 
                 pad = 100, distance=0.1, num_layers=2, amp_factor=6, approx="Fresnel3"):
        super(DiffractiveClassifier_RGB_residual, self).__init__()
        self.amp_factor = amp_factor
        self.size = sys_size
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pad = pad
        self.approx=approx
        self.use_det = use_det
        
        self.num_layers = num_layers
        if batch_norm == True:
            raise NotImplementedError("Batch normalization is not implemented yet.")
        
        
        self.diffractive_layers_r = torch.nn.ModuleList([layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                                                pixel_size=self.pixel_size,
                                                                                size=self.size, 
                                                                                pad = self.pad, 
                                                                                distance=self.distance,
                                                                                amplitude_factor = amp_factor, 
                                                                                approx=self.approx,
                                                                                phase_mod=True) for _ in range(num_layers)])
        self.diffractive_layers_g = torch.nn.ModuleList([layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                                                pixel_size=self.pixel_size,
                                                                                size=self.size, 
                                                                                pad = self.pad, 
                                                                                distance=self.distance,
                                                                                amplitude_factor = amp_factor, 
                                                                                approx=self.approx,
                                                                                phase_mod=True) for _ in range(num_layers)])
        self.diffractive_layers_b = torch.nn.ModuleList([layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                                                pixel_size=self.pixel_size,
                                                                                size=self.size, 
                                                                                pad = self.pad, 
                                                                                distance=self.distance,
                                                                                amplitude_factor = amp_factor, 
                                                                                approx=self.approx,
                                                                                phase_mod=True) for _ in range(num_layers)])
        
        self.last_diffraction_r = layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                         pixel_size=self.pixel_size,
                                                         size=self.size, 
                                                         pad = self.pad, 
                                                         distance=self.distance,
                                                         approx=self.approx, 
                                                         phase_mod=False)
        self.last_diffraction_g = layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                         pixel_size=self.pixel_size,
                                                         size=self.size, 
                                                         pad = self.pad, 
                                                         distance=self.distance,
                                                         approx=self.approx, 
                                                         phase_mod=False)
        self.last_diffraction_b = layers.DiffractLayer_Raw(wavelength=self.wavelength, 
                                                         pixel_size=self.pixel_size,
                                                         size=self.size, 
                                                         pad = self.pad, 
                                                         distance=self.distance,
                                                         approx=self.approx, 
                                                         phase_mod=False)
        
        if self.use_det:
            self.detector = layers.Detector(x_loc=det_x_loc, y_loc=det_y_loc, det_size=det_size, size=self.size)

        self.shortcut = nn.Sequential()
        
    def forward(self, x, no_clamp=False):
        xr, xg, xb = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        
        if self.num_layers == 5 or self.num_layers == 10:
            xr = self.diffractive_layers_r[0](xr)
            xr_skip = self.shortcut(xr)
            for index in range(1, len(self.diffractive_layers_r)):
                if index % 2 == 1:
                    xr = self.diffractive_layers_r[index](xr)
                else:
                    xr = self.diffractive_layers_r[index](xr)
                    xr = (xr + xr_skip) / 2
                    xr_skip = self.shortcut(xr)
                    
            xg = self.diffractive_layers_r[0](xg)
            xg_skip = self.shortcut(xg)
            for index in range(1, len(self.diffractive_layers_g)):
                if index % 2 == 1:
                    xg = self.diffractive_layers_g[index](xg)
                else:
                    xg = self.diffractive_layers_g[index](xg)
                    xg = (xg + xg_skip) / 2
                    xg_skip = self.shortcut(xg)
                    
            xb = self.diffractive_layers_g[0](xb)
            xb_skip = self.shortcut(xb)
            for index in range(1, len(self.diffractive_layers_b)):
                if index % 2 == 1:
                    xb = self.diffractive_layers_b[index](xb)
                else:
                    xb = self.diffractive_layers_b[index](xb)
                    xb = (xb + xb_skip) / 2
                    xb_skip = self.shortcut(xb)
                
        xr = self.last_diffraction_r(xr)
        xg = self.last_diffraction_g(xg)
        xb = self.last_diffraction_b(xb)
        
        x = torch.stack([xr,xg,xb], dim=1)[:, :, 0]
        
        if self.use_det:
            output = self.detector(x)
            return output
        else:
            if no_clamp:
                return x.abs()
            return F.hardtanh(x.abs(), 0, 1)
        
if __name__ == "__main__":    
    model = DiffractiveClassifier_Raw(num_layers=depth, batch_norm=False, device=device,
                                det_x_loc=det_x_loc, det_y_loc=det_y_loc, det_size=det_size,
                                wavelength=wavelength, pixel_size=pixel_size, sys_size=sys_size, pad=pad,
                                distance=distance, amp_factor=amp_factor, approx=approx)