import numpy as np
from operators import grad, wavelet_operator_1d, prox_op, adjoint_wavelet_operator_1d, blur_operator
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
        z = y_old - Linv*grad(y_old, b)
        c = wavelet_operator_1d(z)
        d = prox_op(c[0],lam/Linv)
        x = adjoint_wavelet_operator_1d(d,c[1])
        t = 0.5*(1 + np.sqrt(1 + 4*t_old**2))
        y = x + (t_old/t)*(x - x_old)
        # y = x
        step = abs((y-y_old)/Linv)
        max_step = step.max()
        step_size_list.append(max_step)
        function_values.append((np.linalg.norm(blur_operator(y) - b))**2 + lam*np.linalg.norm(c[0], ord=1))
    end = time.time()
    return y, start, end, step_size_list, function_values

def FISTA_SR3(x,y,b,t,k,max_iter,lam,eta):
    return 0