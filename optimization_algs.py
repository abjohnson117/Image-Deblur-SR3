import numpy as np
import torch
from operators import grad, wavelet_op1d_torch, prox_op, prox_op_torch, wavelet_inverse_torch, blur_operator, gen_function
import time

def FISTA(x,y,b,t,k,max_iter,lam,Linv): #, grad, func, prox
    start = time.time()
    step_size_list = []
    function_values = []
    while (k <= max_iter):
        k += 1
        x_old = x
        y_old = y
        t_old = t
        z = y_old - Linv*grad(y_old, b, gen_function)
        c = wavelet_op1d_torch(z)
        d = prox_op_torch(c[0],lam/Linv)
        x = wavelet_inverse_torch(d,c[1])
        t = 0.5*(1 + np.sqrt(1 + 4*t_old**2))
        y = x + (t_old/t)*(x - x_old)
        # y = x
        step = abs((y-y_old)/Linv)
        max_step = torch.max(step)
        step_size_list.append(max_step)
        # function_values.append((gen_function(y,b) + lam*np.linalg.norm(c[0], ord=1))
    end = time.time()
    return y, start, end, step_size_list, function_values

def FISTA_SR3(x,y,b,t,k,max_iter,lam,eta):
    return 0