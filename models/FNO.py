import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        # x_ft[0] = 0.5 * x_ft[0]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, fno_architecture, device="cpu", nfun=1, padding_frac=1 / 4):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes = fno_architecture["modes"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.retrain_fno = fno_architecture["retrain"]

        torch.manual_seed(self.retrain_fno)
        self.padding_frac = padding_frac
        self.fc0 = nn.Linear(nfun + 1, self.width)
        self.conv_list = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList(
            [SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.to(device)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x_padding = int(round(x.shape[-1] * self.padding_frac))
        x = F.pad(x, [0, x_padding])

        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = F.gelu(x)

        x = x[..., :-x_padding]

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)



class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

#------------------------------------------------------------------------------

class FNO2d(nn.Module):
    def __init__(self, fno_architecture, in_channels = 1, out_channels = 1, device=None):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = fno_architecture["modes"]
        self.modes2 = fno_architecture["modes"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.retrain_fno = fno_architecture["retrain"]
        self.padding = fno_architecture["padding"]
        self.include_grid = fno_architecture["include_grid"]
        self.input_dim = in_channels
        self.act  = nn.LeakyReLU() 
        self.device = device

        torch.manual_seed(self.retrain_fno)
        
        if self.include_grid == 1:
            self.r = nn.Sequential(nn.Linear(self.input_dim+2, 128),
                                   self.act,
                                   nn.Linear(128, self.width))
        else:
            self.r = nn.Sequential(nn.Linear(self.input_dim, 128),
                                   self.act,
                                   nn.Linear(128, self.width))
        
        
        
        self.conv_list = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])

        
        self.q = nn.Sequential(nn.Linear(self.width, 128),
                                self.act,
                                nn.Linear(128, out_channels))
        
        self.to(device)
                
    def get_grid(self, samples, res):
        size_x = size_y = res
        samples = samples
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([samples, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([samples, 1, size_x, 1])
        grid = torch.cat((gridy, gridx), dim=-1)

        return grid

    def forward(self, x):
                
        x = x.permute(0, 2, 3, 1)

        if self.include_grid == 1:
            grid = self.get_grid(x.shape[0], x.shape[1]).to(self.device)
            x = torch.cat((grid, x), -1)
        
        x = self.r(x)
        x = x.permute(0, 3, 1, 2)
        
        x1_padding =  self.padding
        x2_padding =  self.padding
                
        if self.padding>0: 
            x = F.pad(x, [0, x1_padding, 0, x2_padding])

        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = self.act(x)
        
        del x1
        del x2
        
        if self.padding > 0:
            x = x[..., :-x1_padding, :-x2_padding]            
        x = x.permute(0, 2, 3, 1)
        x = self.q(x)

        return x.permute(0, 3, 1, 2)


class FactorizedSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FactorizedSpectralConv2d, self).__init__()

        """
        Factorized 2D Fourier layer (FFNO style).
        Instead of a full 2D FFT and a dense weight tensor, it performs:
        1. FFT on X -> Multiply WeightX -> IFFT on X
        2. FFT on Y -> Multiply WeightY -> IFFT on Y
        3. Sum the results.
        This reduces parameter complexity significantly.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Modes for X dimension
        self.modes2 = modes2 # Modes for Y dimension

        self.scale = (1 / (in_channels * out_channels))
        
        # Weights are separated per dimension
        # Weight for X: (in, out, modes1)
        self.weights_x = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        # Weight for Y: (in, out, modes2)
        self.weights_y = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        # x shape: (Batch, Channels, H, W)
        B, C, H, W = x.shape

        # --- Branch 1: Y Dimension (Last Dim) ---
        # 1. FFT on Y
        x_fty = torch.fft.rfft(x, dim=-1)
        
        # 2. Multiply relevant Fourier modes
        # We need to initialize the output buffer in complex domain
        out_fty = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        # Slice to modes2
        slice_y = x_fty[..., :self.modes2]
        
        # Einsum: (Batch, In, H, ModesY) x (In, Out, ModesY) -> (Batch, Out, H, ModesY)
        # We keep H (dim 2) intact here
        out_fty[..., :self.modes2] = torch.einsum('bixy, ioy -> boxy', slice_y, self.weights_y)
        
        # 3. IFFT on Y
        out_y = torch.fft.irfft(out_fty, n=W, dim=-1)

        # --- Branch 2: X Dimension (Second to Last Dim) ---
        # 1. FFT on X
        x_ftx = torch.fft.rfft(x, dim=-2)
        
        # 2. Multiply relevant Fourier modes
        # Note: rfft on dim -2 results in shape (B, C, H//2+1, W)
        out_ftx = torch.zeros(B, self.out_channels, H // 2 + 1, W, device=x.device, dtype=torch.cfloat)
        
        # Slice to modes1
        slice_x = x_ftx[:, :, :self.modes1, :]
        
        # Einsum: (Batch, In, ModesX, W) x (In, Out, ModesX) -> (Batch, Out, ModesX, W)
        # We keep W (dim 3) intact here
        out_ftx[:, :, :self.modes1, :] = torch.einsum('bixy, iox -> boxy', slice_x, self.weights_x)
        
        # 3. IFFT on X
        out_x = torch.fft.irfft(out_ftx, n=H, dim=-2)

        # --- Combine ---
        return out_x + out_y


class FactorizedFNO2d(nn.Module):
    def __init__(self, fno_architecture, in_channels=1, out_channels=1, device=None):
        super(FactorizedFNO2d, self).__init__()

        """
        The Factorized FNO 2D Network. 
        identical structure to FNO2d, but uses FactorizedSpectralConv2d layers.
        """
        self.modes1 = fno_architecture["modes"]
        self.modes2 = fno_architecture["modes"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.retrain_fno = fno_architecture["retrain"]
        self.padding = fno_architecture["padding"]
        self.include_grid = fno_architecture["include_grid"]
        self.input_dim = in_channels
        self.act = nn.LeakyReLU()
        self.device = device

        torch.manual_seed(self.retrain_fno)

        if self.include_grid == 1:
            self.r = nn.Sequential(nn.Linear(self.input_dim + 2, 128),
                                   self.act,
                                   nn.Linear(128, self.width))
        else:
            self.r = nn.Sequential(nn.Linear(self.input_dim, 128),
                                   self.act,
                                   nn.Linear(128, self.width))

        self.conv_list = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        # USE FACTORIZED LAYER HERE
        self.spectral_list = nn.ModuleList(
            [FactorizedSpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])

        self.q = nn.Sequential(nn.Linear(self.width, 128),
                               self.act,
                               nn.Linear(128, out_channels))

        self.to(device)

    def get_grid(self, samples, res):
        size_x = size_y = res
        samples = samples
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([samples, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([samples, 1, size_x, 1])
        grid = torch.cat((gridy, gridx), dim=-1)
        return grid

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        if self.include_grid == 1:
            grid = self.get_grid(x.shape[0], x.shape[1]).to(self.device)
            x = torch.cat((grid, x), -1)

        x = self.r(x)
        x = x.permute(0, 3, 1, 2)

        x1_padding = self.padding
        x2_padding = self.padding

        if self.padding > 0:
            x = F.pad(x, [0, x1_padding, 0, x2_padding])

        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):
            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = self.act(x)

        if self.padding > 0:
            x = x[..., :-x1_padding, :-x2_padding]
        x = x.permute(0, 2, 3, 1)
        x = self.q(x)

        return x.permute(0, 3, 1, 2)
