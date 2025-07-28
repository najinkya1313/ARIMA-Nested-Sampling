##Python packages
import sys
sys.path.append('/Users/ajinkya/Desktop/Python notebooks/ARIMA-Nested-Sampling')
from ARIMA_fast import ARIMA_fast
import blackjax
import time
import jax.numpy as jnp
from anesthetic import NestedSamples 
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import jax

##Nested Sampler for ARIMA Models:

def ARIMA_Nested_Sampler(data,order,lklhood,prior_bounds,num_live,num_delete):
    """
    A function for Nested Sampling with ARIMA Models using the Blackjax Nested Sampler and returning the processed results and posterior plots.
    Args:
     data (array or list) : The time_series data to be fitted.
     order (tuple) : (p,d,q) order of the ARIMA model.
     lklhood (func) : The log-likelihood function containing the ARIMA model.
     prior_bounds (dict) : Bounds on the prior distribution.
     num_live : number of live points to draw from the prior distribution.
     num_delete : number of points to delete at each iteration.
    """
    num_dims = len(prior_bounds)
    num_inner_steps = num_dims * 5
    if num_dims!=(np.sum(order)+1):
        raise Exception("Number of parameters in prior_bounds inconsistent with ARIMA order.")
    else:
     rng_key = jax.random.PRNGKey(16)
     rng_key,prior_key = jax.random.split(rng_key)
     particles,logprior_fn = blackjax.ns.utils.uniform_prior(prior_key,num_live,prior_bounds)
     ##Nested Sampler
     nested_sampler = blackjax.nss(logprior_fn=logprior_fn,loglikelihood_fn = lklhood,num_delete=num_delete,num_inner_steps=num_inner_steps)
     init_fn = jax.jit(nested_sampler.init)
     step_fn = jax.jit(nested_sampler.step)
     ns_start = time.time()
     live = init_fn(particles)
     dead = []
    
     with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
       while not live.logZ_live - live.logZ < -3:  # Convergence criterion
         rng_key, subkey = jax.random.split(rng_key, 2)
         live, dead_info = step_fn(subkey, live)
         dead.append(dead_info)
         pbar.update(num_delete)
    
     dead = blackjax.ns.utils.finalise(live,dead)
     ns_time = time.time() - ns_start
     ##Processing results
     columns = [i for i in prior_bounds.keys()]
    
     data = jnp.vstack([dead.particles[key] for key in columns]).T

     posterior_samples = NestedSamples(
     data,
     logL=dead.loglikelihood,
     logL_birth=dead.loglikelihood_birth,
     columns=columns,
     labels=None,
     logzero=jnp.nan,
     )
     #Print results:
     print(f"Nested sampling runtime: {ns_time:.2f} seconds")
     print(f"Log Evidence: {posterior_samples.logZ():.2f} ± {posterior_samples.logZ(100).std():.2f}")
     print(type(posterior_samples))
    
    
     # Create posterior corner plot with true values marked
     kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
     axes = posterior_samples.plot_2d(columns, kinds=kinds, label='Posterior')
     
     plt.suptitle("Line Fitting: Posterior Distributions")
     return posterior_samples


    


   
    



        
            
            




        
    
    
    

    


   
    



        
            
            




        
    
    
    
