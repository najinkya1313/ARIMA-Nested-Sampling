import numpy as np
import jax
import jax.numpy as jnp
##Defining an ARIMA function from scratch
#(Removing sigma parameter for the purpose of Nested Sampling)
def ARIMA_custom(data,order:tuple,sigma,phi:tuple,theta:tuple):
 
 "A non-seasonal ARIMA function that returns the forecasted values y_t for a given time-series data"   
 

 p,d,q = order[0],order[1],order[2]
 
 ##Converting inputs into jax numpy arrays
 phi = jnp.atleast_1d(phi)
 theta = jnp.atleast_1d(theta)
 data = jnp.array(data)

 ## Differencing for stationarity
 diff_data = jnp.diff(data,d)
 
 ##Defining the constant intercept k
 phi_sum = jnp.sum(phi)
 k = (1-phi_sum)*jnp.mean(diff_data)
 
 
 ##Padding with zeroes for first forecast
 if p>q:
   diff_data = jnp.concatenate((jnp.array(p*[jnp.mean(data)]),data)) 
 else:
   diff_data = jnp.concatenate((jnp.array(p*[jnp.mean(data)]),data))
 
 ## Autoregression and MA part:
 y_t_diffed = q*[0]
 k = (1-phi_sum)*jnp.mean(diff_data)
 for index,val in enumerate(diff_data):
    if index<(len(diff_data)-p):
        y_t=[]
        iter_var_ar=0
        while iter_var_ar<p:
            y = phi[iter_var_ar]*diff_data[index+iter_var_ar] 
            y_t.append(y)
            iter_var_ar+=1
        
        iter_var_ma = 0
        while iter_var_ma<q:
           error = diff_data[index+iter_var_ma] - y_t_diffed[index+iter_var_ma]
           y_t.append(theta[iter_var_ma]*error)
           iter_var_ma+=1
        y_t = jnp.array(y_t)
        y_t_diffed = list(y_t_diffed)
        y_t_diffed.append(k+jnp.sum(y_t))
        y_t_diffed = jnp.array(y_t_diffed)

 ##Integrating the forecasts to match the original time-series:
 if d==0:
     integrated = y_t_diffed
    
 else:
     integrated = jnp.asarray(y_t_diffed).copy()
     initial_values = data[:d]   
     for i in range(d, 0, -1):
      first_val = jnp.diff(initial_values, n=i-1)[0]
      integrated = jnp.concatenate(([first_val], jnp.cumsum(integrated) + first_val))
 
 return jax.lax.cond(
    sigma == 0,
    lambda _: integrated,
    lambda _: integrated + sigma * jax.random.normal(jax.random.PRNGKey(27), shape=integrated.shape),
    operand=None
   )
