from matplotlib import pyplot as plt

def FISTA_plot(y,m):
    y_fis = y.reshape(m,m, order="F")
    plt.imshow(y_fis,interpolation="nearest",cmap=plt.cm.gray)
    plt.title("Deblurred Image")

def function_vals_plot(max_iter, function_vals):
    max_iter_list = list(range(max_iter))
    plt.plot(max_iter_list,function_vals[:max_iter])
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")

def step_size_plot(max_iter, step_size):
    max_iter_list = list(range(max_iter))
    plt.plot(max_iter_list,step_size[:max_iter])
    plt.xlabel("Iteration")
    plt.ylabel("Step Size")