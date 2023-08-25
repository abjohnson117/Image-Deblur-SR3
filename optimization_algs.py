import numpy as np
import torch
from operators import grad, blur_operator_torch, blur_adjoint_torch
import time

def FISTA(x,y,b,t,k,max_iter,lam,Linv,prox,f,g,accelerate): #, grad, func, prox
    """
    Implementing FISTA algorithm described in Fista paper by Beck and Teboulle
    """
    start = time.time()
    step_size_list = []
    function_values = []
    while (k <= max_iter):
        k += 1
        x_old = x
        y_old = y
        t_old = t
        grd = grad(y_old, b, f)
        with torch.no_grad():
            # Here we wrap with a torch.no_grad() bc the only computations we want to be tracked in the gradient computational graph
            # are those pertaining to my grad() function. Everything else we don't want in our computation graph, and hence,
            # it's inside the with torch.no_grad() block.
            z = y_old - Linv*grd
            x = prox(z,lam/Linv)
            if accelerate:
                t = 0.5*(1 + np.sqrt(1 + 4*t_old**2))
                y = x + (t_old/t)*(x - x_old)
            else:
                y = x
            step = abs((y-y_old)/Linv)
            max_step = torch.max(step)
            step_size_list.append(max_step)
            function_values.append((f(y,b) + g(lam,x)))
        y_old.grad.zero_()
    end = time.time()
    return y, start, end, step_size_list, function_values

def FISTA_SR3(w,v,b,t,k,max_iter,eta,prox,kappa,lam,m,sr3_function,it_num=20,accelerate=True):
    """
    Implenting SR3 with Fista acceleration just as in J33 paper. No TV regularization,
    we regularize with 1-norm with C = I.
    """
    start = time.time()
    step_size_list = []
    function_values = []
    atb = blur_adjoint_torch(b)
    x_init = torch.zeros(m**2)
    while (k <= max_iter):
        k += 1
        v_old = v
        w_old = w
        t_old = t
        z = atb + kappa*w_old
        x = conjgrad(H_kappa,z,x_init,it_num,kappa)
        v = prox(x,eta*lam)
        if accelerate:
            t = 0.5*(1 + np.sqrt(1 + 4*(t_old**2)))
            w = v + (t_old/t)*(v - v_old)
        else:
            w = v
        step = abs((w-w_old)*kappa)
        max_step = torch.max(step)
        function_values.append(sr3_function(x,w,b,eta))
        step_size_list.append(max_step)
    end = time.time()

    return x,w, start, end, step_size_list, function_values


def conjgrad(op, b, x, it_num, kap):
    """
    Simple CG-algorithm implemented in PyTorch
    """
    r = b - op(kap,x)
    p = r
    rsold = torch.dot(r, r)  # Calculate dot product for rsold

    for _ in range(it_num):
        Ap = op(kap,p)
        alpha = rsold / torch.dot(p, Ap)  # Calculate dot product for alpha
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)  # Calculate dot product for rsnew

        if torch.sqrt(rsnew) < 1e-10:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

def H_kappa(kap, x):
    """
    The H_k matrix operation defined in J33 in the context of image deblurring
    with blur and blur_adjoint operations defined for PyTorch tensors.
    """
    return blur_adjoint_torch(blur_operator_torch(x)) + kap*x