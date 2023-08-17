import numpy as np
import torch
from operators import prox_op, adjoint_prox_op, wavelet_op1d_torch, wavelet_inverse_torch, grad_check, grad, blur_operator_torch, blur_adjoint_torch, gen_function

def torch_arranger(x,b):
    """
    This is boiler code to simply get everything in the torch format we need for the tests to work.
    """
    if type(x) != torch.Tensor:
        x = torch.from_numpy(x)
    if type(b) != torch.Tensor:
        b = torch.from_numpy(b)
    if len(x.size()) > 1:
        x = torch.flatten(x)
    if len(b.size()) > 1:
        b = torch.flatten(b)
    return x,b

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
    x,b = torch_arranger(x,b)
    bsss = wavelet_op1d_torch(b)[1]
    error = torch.matmul(wavelet_op1d_torch(x)[0],b) - torch.matmul(x,wavelet_inverse_torch(b,bsss))
    if error < thresh:
        return True
    else:
        print("This test fails")
        return torch.matmul(wavelet_op1d_torch(x)[0],b), torch.matmul(x,wavelet_inverse_torch(b,bsss)), error

def test_grad(x,b,thresh=1e-4):
    """
    From the way we are creating and using the grad function from PyTorch AutoDiff, we are not calculating adjoints.
    Instead, we simply compare torch's autodiff output to what we believe it should be as it is hardcoded.
    """
    if type(x) != torch.Tensor:
        x = torch.from_numpy(x)
    if type(b) != torch.Tensor:
        b = torch.from_numpy(b)
    if len(x.size()) > 1:
        x = torch.flatten(x)
    if len(b.size()) > 1:
        b = torch.flatten(b)
    error = torch.linalg.norm(grad(x,b,gen_function) - grad_check(x,b))
    if error < thresh:
        return True
    else:
        print("This test fails")
        return grad(x,b,gen_function), grad_check(x,b), error

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