import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1,))          # ‘order’ is static
def ARIMA_fast(data,order,sigma,mu, phi,theta,init_y,init_e, seed):
    """
    Vectorised non-seasonal ARIMA(p,d,q) for JAX/XLA.
    """
    key = jax.random.PRNGKey(seed)
    key,arima_key = jax.random.split(key)
    p, d, q = order
    x_dtype = jnp.asarray(data).dtype            # keep original dtype
    data    = jnp.asarray(data, dtype=jnp.float32)  # or x_dtype

    # 1. Differencing -------------------------------------------------------------
    diff = jnp.diff(data, n=d) if d else data

    # 2. Parameters / intercept ---------------------------------------------------
    phi   = jnp.pad(jnp.asarray(phi,   dtype=diff.dtype), (0, p - len(phi)))
    theta = jnp.pad(jnp.asarray(theta, dtype=diff.dtype), (0, q - len(theta)))
    k = mu * (1- jnp.sum(phi))
    

    # 3. Initial state ------------------------------------------------------------
    past_y = jnp.array(init_y) if p else jnp.empty((0,), diff.dtype)
    past_e = jnp.array(init_e) if q else jnp.empty((0,), diff.dtype)

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
        lambda r: r + sigma * jax.random.normal(arima_key, r.shape, dtype=r.dtype),
        recovered,
    )

    return recovered


def ARIMA_forecast(data,order,sigma,mu,phi,theta,forecast_num,seed):
    r"""A function for forecasting future values for a given time-series data (can also be used for generating artificial ARIMA data) 
    Args:
     data : time-series data to use for forecasting
     order : (p,d,q) order of the ARIMA Model
     sigma : $\sigma$ value for present shocks $\epsilon_t$
     phi : list or array of the AR $\phi$ coefficients of ARIMA Model
     theta : list or array of the MA $\theta$ coefficients of ARIMA Model
     forecast_num : number of future points to forecast
     seed : seed for random number generator
    
    
    """
    y_model = ARIMA_fast(data,order,sigma,mu,phi,theta,seed)
    p,d,q = order
    phi_coeffs = jnp.array(phi)
    theta_coeffs = jnp.array(theta)
    forecasted_points = []
    rng_key = jax.random.PRNGKey(seed)
    error_key = jax.random.split(rng_key,forecast_num)
    
    k = mu * (1- jnp.sum(phi_coeffs))
    while len(forecasted_points)<forecast_num:
        epsilon_lagged = data[-q:] - y_model[-q:]
        for key in error_key:
            if p:
                y_phis = phi_coeffs*jnp.flip(data[-p:])
            if q==0:
                epsilon_lagged = jnp.empty(q)
            if p==0:
                y_phis = jnp.empty(p)
            y_thetas = theta_coeffs * jnp.flip(epsilon_lagged)
            epsilon_t = sigma * jax.random.normal(key)
            y_forecast = k + jnp.sum(y_phis) + jnp.sum(y_thetas) + epsilon_t
            forecasted_points.append(y_forecast)
            y_forecast_arr = jnp.array([y_forecast])
            data = jnp.concatenate([data,y_forecast_arr])
            y_model = jnp.concatenate([y_model,y_forecast_arr])
            
    
    
    forecasted_points = jnp.array(forecasted_points)
    return forecasted_points


        
