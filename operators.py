import numpy as np
import pywt
from scipy.ndimage import correlate, convolve
from scipy.signal import convolve2d
from scipy.fftpack import dct
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from pytorch_wavelets import DWT1DForward, DWT1DInverse

def prox_op(x,lambd):
    return np.sign(x)*np.maximum(np.abs(x)-lambd,0)

def prox_op_torch(x,lambd):
    return torch.sign(x) * torch.relu(torch.abs(x) - lambd)

def adjoint_prox_op(x,lambd):
    return x - prox_op(x,lambd)

def wavelet_operator_1d(org, mode="reflect"):
    coeffs = pywt.wavedec(org, wavelet="haar", level=2, mode=mode)
    wav_x, keep = pywt.coeffs_to_array(coeffs)
    return wav_x, keep

def wavelet_op1d_torch(org, mode="reflect"):
    """
    Use pytorch wavelets instead of pywt
    link: https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html
    """
    dwt = DWT1DForward(J=3, wave="haar", mode=mode)
    wav_x, coeffs = dwt(org.unsqueeze(0).unsqueeze(0))
    wav_x = wav_x.squeeze().squeeze()
    coeff_list = [coeff.squeeze().squeeze() for coeff in coeffs]
    return wav_x, coeff_list

def wavelet_inverse_torch(wav_x, coeff_list, mode="reflect"):
    """
    Use pytorch wavelets instead of pywt to do inverse transformation
    link: https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html
    """
    wav_x = wav_x.unsqueeze(0).unsqueeze(0)
    coeffs = [coeff.unsqueeze(0).unsqueeze(0) for coeff in coeff_list]
    idwt = DWT1DInverse(mode=mode, wave="haar")
    org = idwt((wav_x,coeffs))
    org = org.squeeze().squeeze()
    return org

def adjoint_wavelet_operator_1d(wav_x, keep, mode="reflect"):
    coeffs = pywt.array_to_coeffs(wav_x, keep, output_format='wavedec')
    org = pywt.waverec(coeffs, wavelet="haar", mode=mode)
    return org


def fspecial(shape=(3,3),sigma=0.5, ret_torch=False):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    if ret_torch:
        h = torch.from_numpy(h)
        h = h.to(torch.float)
    return h

def blur_operator(org, reshape=True, shape=(9,9), sigma=4, mode="reflect"):
    # if type(org) == torch.Tensor:
    #     org = org.numpy()
    if reshape:
        m = int(np.sqrt(org.shape[0]))
        org = np.reshape(org, (m,m), order="F")
        # org = org.view(m,m)

    psf = fspecial(shape,sigma)
    blurred = correlate(org, psf, mode=mode)
    # blurred += np.random.normal(0,1e-4,size=(m,m)) #TODO: Add this separately
    blurred = blurred.T
    if reshape:
        blurred = blurred.flatten("F")
    
    return torch.from_numpy(blurred)

def blur_operator_torch(org, shape=(9,9), sigma=4.0):
    if type(org) != torch.Tensor:
        org = torch.from_numpy(org)

    if len(org.size()) == 1:
        m = int(np.sqrt(org.size(0)))
        org = torch.reshape(org, (m,m))
    
    org = org.unsqueeze(0)
    blurrer = GaussianBlur(shape, sigma) #https://pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html
    blurred = blurrer(org)
    blurred = blurred.squeeze()
    blurred = torch.flatten(blurred)

    return blurred

def blur_adjoint_torch(org, shape=(9,9), sigma=4, mode="reflect"):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html Look at this link for convolution transpose.
    I think the idea might be to get the Gaussian Blur kernel from GaussianBlur and use it as the kernel in the convolution transpose.
    """
    if type(org) != torch.Tensor:
        org = torch.from_numpy(org)

    if len(org.size()) == 1:
        m = int(np.sqrt(org.size(0)))
        org = torch.reshape(org, (m,m))

    psf = fspecial(shape, sigma, True)
    psf = torch.flip(psf, [0,1])
    padding = nn.ReflectionPad2d(4)
    org.unsqueeze_(0).unsqueeze_(0)
    org_pad = padding(org)
    org_pad = org_pad.to(torch.float)
    adjoint_blurred = F.conv2d(org_pad, psf.unsqueeze(0).unsqueeze(0))
    adjoint_blurred = adjoint_blurred.squeeze()
    adjoint_blurred = torch.flatten(adjoint_blurred)

    return adjoint_blurred

def blur_adjoint(org, reshape=True, shape=(9,9), sigma=4, mode="reflect"):
    if reshape:
        m = int(np.sqrt(org.shape[0]))
        org = np.reshape(org, (m,m), order="F")

    psf = fspecial(shape, sigma)
    psf = np.flipud(np.fliplr(psf))
    adjoint_blurred = correlate(org, psf, mode=mode)
    adjoint_blurred = adjoint_blurred.T
    if reshape:
        adjoint_blurred = adjoint_blurred.flatten("F")

    return adjoint_blurred

def dctshift(psf, center=(4,4)):
    """Taken from Deblurring Images to compute first column of A  matrix"""
    m, n = psf.shape[0], psf.shape[1]
    i = center[0]
    j = center[1]
    k = min([i,m-i,j,n-j])

    PP = psf[i-(k):i+(k+1),j-(k):j+(k+1)]
    Z1 = np.diag(np.ones(k+1), k)
    Z2 = np.diag(np.ones(k), k+1)

    PP = Z1@PP@Z1.T + Z1@PP@Z2.T + Z2@PP@Z1.T + Z2@PP@Z2.T 
    Ps = np.zeros((m,n))
    Ps[0:2*k+1,0:2*k+1] = PP

    return Ps

def evals_blur(psf):
    """Calculates eigenvalues according to equation in Deblurring Images"""
    a1 = dctshift(psf)
    
    e1 = np.zeros_like(a1)
    e1[0,0] = 1

    S = dct(dct(a1, axis=0), axis=1) / dct(dct(e1,axis=0), axis=1)
    return np.max(S), S

def gen_function(x,b):
    Ax = blur_operator_torch(x)
    w = Ax - b
    return (torch.linalg.norm(w)**2)

def grad(x, b, fcn):
    """
    Uses PyTorch autodifferentiation to compute gradients at every step
    """
    if type(b) != torch.Tensor:
        b = torch.from_numpy(b)
    if type(x) != torch.Tensor:
        x = torch.from_numpy(x)
    if x.requires_grad == False:
        x.requires_grad_()
    w = fcn(x,b)
    w.backward()
    return x.grad
