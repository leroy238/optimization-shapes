import numpy as np

N = 100_000
d = 2
epochs = 10_000
beta = 0 # PLACEHOLDER

def f(x, c, p):
    return np.sum(c.repeat(1, x.shape[0]) @ (x ** p) - c)
#end for

def grad_f(x, c, p):
    pass
#end grad_f

def hess_f(x, c, p):
    pass
#end hess_f

def gd_update(x, L, c, p):
    return x - 1/L * grad_f(x, c, p)
#end gd_update

def sgd_update(x, L, c, p):
    # Note that here c is sampled, not the full dataset.
    return x - 1/L * grad_f(x, c, p)
#end sgd_update

def nag_update(x, L, c, p):
    pass
#end nag_update

def sgd_momentum_update(x, L, z, c, p):
    z = beta * z + (1 - beta) * 1 / L * grad_f(x, c, p)
    return z, x - z
#end sgd_momentum_update

def adam_update(x, L, m, v, c, p):
    pass
#end adam_update

def newton_update(x, c, p):
    return x - np.linalg.inv(hess_f(x, c, p)) @ grad_f(x, c, p)
#end newton_update

def train(x_init, update, dataset, L, epochs):
    x = x_init.copy()
    
    # Placeholders such that the shape is the same
    z = x_init.copy()
    m = x_init.copy()
    v = x_init.copy()
    
    error = []
    
    if update != sgd_update and update != sgd_momentum_update:
        # Not stochastic
        for epoch in epochs:
            if update != adam_update:
                # No memory.
                    if update != newton_update:
                        # No adaptive learning rate
                        x = update(x, L, dataset, p)
                    else:
                        # Adaptive learning rate
                        x = update(x, dataset, p)
                    #end if/else
            else:
                # Momentum
                m, v, x = update(x, L, m, v, dataset, p)
            #end if/else
            
            error.append(f(x, dataset, p))
        #end for
    else:
        # Stochastic
        for epoch in epochs:
            c = None
            if update == sgd_update:
                # No momentum
                x = update(x, L, c, p)
            else:
                # Momentum
                z, x = update(x, L, c, p)
            #end if/else
            
            error.append(f(x, dataset, p))
        #end for
    #end if/else
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