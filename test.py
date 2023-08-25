import jax.numpy as jnp
from jax import grad, vmap, jit
from time import process_time

def metric_tensor(c: jnp.ndarray, data: jnp.ndarray, sigma: float, rho: float):
    sigma2 = sigma ** 2
    D, N = c.shape

    delta = data[:, None, :] - c.T[None, :, :]
    delta2 = delta ** 2
    dist2 = jnp.sum(delta2, axis=2, keepdims=True)
    wn = jnp.exp(-0.5 * dist2 / sigma2)
    
    s = jnp.sum(delta2 * wn, axis=0).T + rho
    m = 1/s
    M = m.T

    return M

N = 5
X = jnp.array(np.random.randn(1000,3))
c = jnp.array(np.random.randn(3,100))
sigma = 0.05
rho = 1e-5

# Using JAX's grad to get the gradient function
gradient_wrt_c = jit(grad(metric_tensor, argnums=0))

t1_start = process_time()
for i in range(N):
    gradient_wrt_c(c, X, sigma, rho)
t1_stop = process_time()

print("Elapsed time for JAX in seconds:", t1_stop-t1_start)
