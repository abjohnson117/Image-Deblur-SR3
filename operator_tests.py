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

def torch_arranger_one(b):
    if type(b) != torch.Tensor:
        b = torch.from_numpy(b)
    if len(b.size()) > 1:
        b = torch.flatten(b)
    return b

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


def test_wavelet(b,thresh=1e-4):
    """
    Adjoint test for our wavelet operator and its inverse. Since the wavelet transform
    creates an orthonormal basis, we know that its inverse is its adjoint. Thus, to confirm
    that the operators pass the adjoint test, we need only see if the inverse function returns
    something almost identical to the original signal passed in.
    """
    b = torch_arranger_one(b)
    wav_b, bsss = wavelet_op1d_torch(b)
    wav_b_inv = wavelet_inverse_torch(wav_b, bsss)
    error = torch.linalg.norm(b - wav_b_inv)
    assert error < thresh, f"Adjoint test for wavelets failed. The normed error is {error:.6f}"

def test_grad(x,b,fcn,thresh=1e-4):
    """
    From the way we are creating and using the grad function from PyTorch AutoDiff, we are not calculating adjoints.
    Instead, we simply compare torch's autodiff output to what we believe it should be as it is hardcoded.
    """
    x, b = torch_arranger(x,b)
    error = torch.linalg.norm(grad(x,b,fcn) - grad_check(x,b))
    x.grad.zero_()
    assert error < thresh, f"Adjoint test for gradient failed. The normed error is {error:.6f}"

def test_blur(x,b,thresh=1e-4):
    """
    This is the unit test for the blur operator. Since blurring is linear, we simply perform a standard
    adjoint test.
    """
    x, b = torch_arranger(x,b)
    error = torch.matmul(blur_operator_torch(x),b) - torch.matmul(x, blur_adjoint_torch(b))
    assert error < thresh, f"Adjoint test for blur operator failed. The error is {error:.6f}"