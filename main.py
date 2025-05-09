import numpy as np
import pickle
import os

N = 100_000
d = 2
p = 1/3
epochs = 1000
beta = 0.99
beta_2 = 0.99

a = None
def init_dataset():
    global a
    path = os.path.join(os.getcwd(), "dataset.pkl")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            a = pickle.load(f)
        #end with
    else:
        a = np.random.normal(loc = 1, scale = 0.2, size = N)
        with open(path, "wb") as f:
            pickle.dump(a, f)
        #end with
    #end if/else
#end init_dataset

def save_data(xs, err, opt = "adam"):
    path = os.path.join(os.getcwd(), f"xs_{opt}.pkl")
    path_err = os.path.join(os.getcwd(), f"error_{opt}.pkl")
    
    with open(path, "wb") as f:
        pickle.dump(xs, f)
    #end with
    
    with open(path_err, "wb") as f:
        pickle.dump(err, f)
    #end with
#end save_data

def raise_power(x, p):
    return (np.abs(x) ** p) * np.sign(x)
#end raise_power

def f(x, c, p):
    x_p = raise_power(x, p)
    return 1/(2 * N) * np.sum((c * np.prod(x_p) - 1)**2)
#end for

def grad_f(x, c, p):
    x_p = raise_power(x, p)
    x_prod = p * np.prod(x_p) / x
    first = np.expand_dims(c * np.prod(x_p) - 1, axis = 1).repeat(2, axis = 1)
    second = np.expand_dims(c, axis = 1) @ np.expand_dims(x_prod, axis = 0)
    
    return 1 / c.shape[0] * np.sum(first * second, axis = 0)
#end grad_f

def hess_f(x, c, p):
    d = x.shape[0]
    x_t = np.expand_dims(x, axis = 1)
    x_p = raise_power(x, p)
    x_prod = np.prod(x_p)
    x_prod_1 = np.expand_dims(p * x_prod / x, axis = 1)
    x_prod_2 = p ** 2 * x_prod / (x_t @ x_t.T) * (np.ones(d,d) - np.diag(np.ones(d)) + np.diag((p - 1) / p * np.ones(d)))
    loss = 0
    for c_i in c:
        loss += c_i * (prod_1 @ prod_1.T + (c_i * prod - 1) * prod_2)
    #end for
    return 1 / c.shape[0] * loss
#end hess_f

def gd_update(x, gamma, z, v, c, p, t):
    return x - gamma * grad_f(x, c, p), z, v
#end gd_update

def nag_update(x, gamma, z, v, c, p, t):
    grad = grad_f(x, c, p)
    z = z - gamma * (t + 1) * grad
    y = x - gamma * grad
    x = (t+1)/(t+3) * y + 2 / (t+3) * z
    return x, z, v
#end nag_update

def gd_momentum_update(x, gamma, z, v, c, p, t):
    z = beta * z + (1 - beta) * gamma * grad_f(x, c, p)
    return x - z, z, v
#end sgd_momentum_update

def adam_update(x, gamma, z, v, c, p, t):
    grad = grad_f(x, c, p)
    z = beta * z + (1 - beta) * grad
    v = beta_2 * v + (1 - beta_2) * grad ** 2
    x = x - gamma * v ** (-1/2) * z
    return x, z, v
#end adam_update

def newton_update(x, gamma, z, v, c, p, t):
    return x - np.linalg.inv(hess_f(x, c, p)) @ grad_f(x, c, p), z, v
#end newton_update

def train(x_init, update, dataset, gamma, epochs, stochastic = True):
    x = x_init.copy()
    
    # Placeholders such that the shape is the same
    z = np.zeros(*x.shape)
    
    if update == nag_update:
        z = x.copy()
    #end if
    
    v = np.zeros(*x.shape)
    
    error = [f(x, dataset, p)]
    xs = [x.copy()]
    
    for epoch in range(epochs):
        if stochastic:
            c = np.random.choice(dataset, size = 1)
        else:
            c = dataset
        #end if/else
        
        x, z, v = update(x, gamma, z, v, c, p, epoch)
    
        error.append(f(x, dataset, p))
        xs.append(x.copy())
    #end for
    
    save_data(xs, error, "sgd" if update != adam_update else "adam")
    
    return error, x
#end train

def main():
    init_dataset()
    dataset = a
    
    gammas = [10 ** x / 10000 for x in range(3, 4)]
    #x_init = np.random.randn(d)
    x_best = np.array([np.mean(1/a)**(1/(2*p)),np.mean(1/a) ** (1/(2*p))])
    x_init = np.array([1.5 ** (1 / p),1.5 ** (1 / p)])
    #best = [-float("inf")]
    for gamma in gammas:
        error, x_star = train(x_init, adam_update, dataset, gamma, epochs, stochastic = True)
        #if min(best) > min(error):
        #    best = error
        #end if
        error, x_star_sgd = train(x_init, gd_update, dataset, gamma, epochs, stochastic = True)
    #end for
    
    #print(error)
    print(x_star, x_star_sgd, np.prod(raise_power(x_star, p)) * np.mean(a))
    print(x_init, x_best, np.mean(a))
#end main

if __name__ == "__main__":
    main()
#end if