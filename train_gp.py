import jax
import jax.numpy as jnp
from jax import random, vmap, value_and_grad, jit
from jax.tree_util import tree_flatten, tree_unflatten

import jaxopt
from jaxopt import ScipyMinimize

#tinygp
from tinygp import kernels, transforms, GaussianProcess

import matplotlib
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)


def get_params_init(key,x0):
  params0 = {
            "theta_c": jnp.ones(()),
            "theta_k": jnp.ones(x0.shape[0]),
  }

  params_rnd = params0.copy()
  for p in params0:
    params_rnd[p] = jnp.log(random.uniform(key,jnp.array(params0[p]).shape,minval=1E-6,maxval=10.))
    _,key = random.split(key)
  
  return params_rnd,key
  return (Xtr,ytr),(Xval,yval)

def data_ch4():
    X = jnp.load("./Data/ch4_geom.npy")
    y = jnp.load("./Data/ch4_energy.npy")
    # gX = jnp.load("./Data/ch4_grad.npy")
    return X, y
  
def build_gp(params, X):
    params = jax.tree_map(lambda x: jnp.exp(x), params)

    mean = 0. # params["mean"]
    diag = 1E-10 # params["diag"] ** 2 

    # Construct the kernel by multiplying and adding `Kernel` objects
    kernel = params["theta_c"] * kernels.ExpSquared(params["theta_k"])

    return GaussianProcess(kernel, X)

def get_data_permutation(key,N):
  X,y = data_ch4()
  i0 = jnp.arange(0,y.shape[0])
  i = random.permutation(key,i0)
  itr = i[:N]
  ival = i[N:]

  ytr = jnp.take(y,itr)
  Xtr = jnp.take(X,itr,axis=0)
  yval = jnp.take(y,ival)
  Xval = jnp.take(X,ival,axis=0)
  return (Xtr,ytr),(Xval,yval)

def neg_log_likelihood(theta, X, y):
    gp = build_gp(theta, X)
    return -gp.condition(y)

  
# main  
N = 1000
_,key = random.split(key)
(Xtr,ytr),(Xval,yval) = get_data_permutation(key,N)
print("Trainin data= ", ytr.shape)
params_init,key = get_params_init(key,Xtr[0])
print(params_init)
print(f"Initial negative log likelihood: {neg_log_likelihood(params_init, Xtr, ytr)}")

# `jax` can be used to differentiate functions, and also note that we're calling
# `jax.jit` for the best performance.
obj = jax.jit(jax.value_and_grad(neg_log_likelihood))

opt = ScipyMinimize(method="BFGS",fun=neg_log_likelihood)
params_opt = opt.run(params_init,Xtr,ytr)
print(params_opt)
print(jax.tree_map(lambda x: jnp.exp(x), params_opt[0]))
print(neg_log_likelihood(params_opt[0], Xtr, ytr))
