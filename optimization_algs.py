import numpy as np
import torch
from operators import grad, wavelet_op1d_torch, prox_op, prox_op_torch, wavelet_inverse_torch, blur_operator, gen_function
import time

def FISTA(x,y,b,t,k,max_iter,lam,Linv,prox, f): #, grad, func, prox
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
            c = wavelet_op1d_torch(z)
            d = prox(c[0],lam/Linv)
            x = wavelet_inverse_torch(d,c[1])
            t = 0.5*(1 + np.sqrt(1 + 4*t_old**2))
            y = x + (t_old/t)*(x - x_old)
            # y = x
            step = abs((y-y_old)/Linv)
            max_step = torch.max(step)
            step_size_list.append(max_step)
            function_values.append((gen_function(y,b) + lam*torch.linalg.norm(c[0], ord=1)))
        y_old.grad.zero_()
    end = time.time()
    return y, start, end, step_size_list, function_values, k

def FISTA_SR3(x,y,b,t,k,max_iter,lam,eta):
    return 0