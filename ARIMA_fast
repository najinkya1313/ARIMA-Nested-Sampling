import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1,))          # ‘order’ is static
def ARIMA_fast(data, order, sigma, phi, theta, *, key=jax.random.PRNGKey(0)):
    """
    Vectorised non-seasonal ARIMA(p,d,q) for JAX/XLA.
    """
    
    sigma = float(sigma)
    p, d, q = order
    x_dtype = jnp.asarray(data).dtype            # keep original dtype
    data    = jnp.asarray(data, dtype=jnp.float32)  # or x_dtype

    # 1. Differencing -------------------------------------------------------------
    diff = jnp.diff(data, n=d) if d else data

    # 2. Parameters / intercept ---------------------------------------------------
    phi   = jnp.pad(jnp.asarray(phi,   dtype=diff.dtype), (0, p - len(phi)))
    theta = jnp.pad(jnp.asarray(theta, dtype=diff.dtype), (0, q - len(theta)))

    k = (1.0 - phi.sum()) * jnp.mean(diff)

    # 3. Initial state ------------------------------------------------------------
    past_y = jnp.full((p,), diff.mean(), dtype=diff.dtype) if p else jnp.empty((0,), diff.dtype)
    past_e = jnp.zeros((q,),                 diff.dtype)   if q else jnp.empty((0,), diff.dtype)

    # 4. One scan step ------------------------------------------------------------
    def one_step(carry, x):
        past_y, past_e = carry

        y_hat = k
        if p:
            y_hat += (phi * past_y).sum()
        if q:
            y_hat += (theta * past_e).sum()

        err = x - y_hat

        if p:
            past_y = jnp.concatenate([jnp.array([x], diff.dtype), past_y[:-1]])
        if q:
            past_e = jnp.concatenate([jnp.array([err], diff.dtype), past_e[:-1]])

        return (past_y, past_e), y_hat

    # 5. Run the recurrence -------------------------------------------------------
    (_, _), y_hat_seq = jax.lax.scan(one_step, (past_y, past_e), diff)

    # 6. Undo differencing --------------------------------------------------------
    if d:
        recovered = jnp.concatenate(
            [data[:d], jnp.cumsum(y_hat_seq) + data[d-1]]
        )
    else:
        recovered = y_hat_seq

    # 7. Optional noise -----------------------------------------------------------
    recovered = jax.lax.cond(
        sigma == 0,
        lambda r: r,
        lambda r: r + sigma * jax.random.normal(key, r.shape, dtype=r.dtype),
        recovered,
    )

    return recovered
