import numpy as np
import torch
from operators import grad, wavelet_op1d_torch, wavelet_inverse_torch, blur_operator_torch, blur_adjoint_torch
import time

def FISTA(x,y,b,t,k,max_iter,lam,Linv,prox,f,accelerate): #, grad, func, prox
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
            c = wavelet_op1d_torch(z) # include the wavelet operation in the prox
            d = prox(c[0],lam/Linv)
            x = wavelet_inverse_torch(d,c[1])
            if accelerate:
                t = 0.5*(1 + np.sqrt(1 + 4*t_old**2))
                y = x + (t_old/t)*(x - x_old)
            else:
                y = x
            step = abs((y-y_old)/Linv)
            max_step = torch.max(step)
            step_size_list.append(max_step)
            function_values.append((f(y,b) + lam*torch.linalg.norm(c[0], ord=1))) # Come up with g function as well
        y_old.grad.zero_()
    end = time.time()
    return y, start, end, step_size_list, function_values

def FISTA_SR3(w,v,b,t,k,max_iter,eta,prox,kappa,m):
    start = time.time()
    step_size_list = []
    # function_values = []
    atb = blur_adjoint_torch(b)
    while (k <= max_iter):
        k +=1
        v_old = v
        w_old = w
        t_old = t
        z = atb + kappa*w_old
        # meta = blur_adjoint_torch(blur_operator_torch(z))
        # meta_inv = torch.linalg.inv(meta.view(m,m))
        x = blur_adjoint_torch(blur_operator_torch(z)) + (1/kappa)*z
        # meta_inv = torch.flatten(meta_inv)
        # x = meta_inv + (1/kappa)*z
        c = wavelet_op1d_torch(x)
        y = prox(c[0],eta)
        v = wavelet_inverse_torch(y,c[1])
        t = 0.5*(1 + np.sqrt(1 + 4*t_old**2))
        w = v + (t_old/t)*(v - v_old)
        step = abs((w-w_old)*kappa)
        max_step = torch.max(step)
        step_size_list.append(max_step)
    end = time.time()

    return w, start, end, step_size_list