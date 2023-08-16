import numpy as np
import torch
from operators import prox_op, adjoint_prox_op, wavelet_operator_1d, adjoint_wavelet_operator_1d, blur_operator, blur_adjoint, grad, blur_operator_torch, blur_adjoint_torch

def test_prox(x, b, lam,thresh=1e-4):
    if len(x.shape) > 1:
        x = x.flatten("F")
    if len(b.shape) > 1:
        b = b.flatten("F")
    error = np.dot(prox_op(x, lam), b) - np.dot(x, adjoint_prox_op(b, lam))
    if np.abs(error) < thresh:
        return True
    else:
        print("This test fails")
        return np.dot(prox_op(x, lam), b), np.dot(x, adjoint_prox_op(b, lam)), error


def test_wavelet(x,b,thresh=1e-4):
    if len(x.shape) > 1:
        x = x.flatten("F")
    if len(b.shape) > 1:
        b = b.flatten("F")
    bsss = wavelet_operator_1d(b)[1]
    error = np.dot(wavelet_operator_1d(x)[0],b) - np.dot(x,adjoint_wavelet_operator_1d(b,bsss))
    if error < thresh:
        return True
    else:
        return False

def test_grad(x,b,thresh=1e-4):
    if len(x.shape) > 1:
        x = x.flatten("F")
    if len(b.shape) > 1:
        b = b.flatten("F")
    error = np.dot(grad(x,b),b) - np.dot(x, adjoint_grad(x,b))
    if error < thresh:
        return True
    else:
        return False

def test_blur(x,b,thresh=1e-4):
    if type(x) != torch.Tensor:
        x = torch.from_numpy(x)
    if type(b) != torch.Tensor:
        b = torch.from_numpy(b)
    if len(x.size()) > 1:
        x = torch.flatten(x)
    if len(b.size()) > 1:
        b = torch.flatten(b)
    error = torch.matmul(blur_operator_torch(x),b) - torch.matmul(x, blur_adjoint_torch(b))
    if error < thresh:
        return True
    else:
        print("This test fails")
        return torch.matmul(blur_operator_torch(x),b), torch.matmul(x, blur_adjoint_torch(b)), error