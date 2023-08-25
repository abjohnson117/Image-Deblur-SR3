from matplotlib import pyplot as plt

def FISTA_plot(y,m,title):
    y_fis = y.view(m,m)
    plt.imshow(y_fis,interpolation="nearest",cmap=plt.cm.gray)
    plt.title(title)

def function_vals_plot(max_iter, function_vals, title):
    max_iter_list = list(range(max_iter))
    plt.plot(max_iter_list,function_vals[:max_iter],label=title)
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.legend(loc="upper right")
    plt.title("Function values plot: " + title)

def step_size_plot(max_iter, step_size, title):
    max_iter_list = list(range(max_iter))
    plt.plot(max_iter_list,step_size[:max_iter],label=title)
    plt.xlabel("Iteration")
    plt.ylabel("Step Size")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.title("Step size plot: " + title)