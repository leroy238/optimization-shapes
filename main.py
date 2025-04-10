import numpy as np

N = 100_000
d = 2

def f(x, c, p):
    return np.sum(c.repeat(1, x.shape[0]) @ (x ** p) - c)
#end for

def grad_f(x, c, p):
    pass
#end grad_f

def hess_f(x, c, p):
    pass
#end hess_f

def gd_update(x, alpha, c, p):
    return x - alpha * grad_f(x, c, p)
#end gd_update

def sgd_update(x, alpha, c, p):
    # Note that here c is sampled, not the full dataset.
    return x - alpha * grad_f(x, c, p)
#end sgd_update

def nag_update(x, L, c, p):
    pass
#end nag_update

def sgd_momentum_update(x, alpha, z, c, p):
    pass
#end sgd_momentum_update

def adam_update(x, alpha, m, v, c, p):
    pass
#end adam_update

def newton_update(x, c, p):
    pass
#end newton_update

def train(x_init, update, dataset, L):
    pass
#end train

def main():
    dataset = np.random.normal(loc = 1, scale = 0.5, size = N)
    
    L = 0 # PLACEHOLDER
    x_init = np.random.rand(d) # PLACEHOLDER
    train(x_init, gd_update, dataset, L)
#end main

if __name__ == "__main__":
    main()
#end if