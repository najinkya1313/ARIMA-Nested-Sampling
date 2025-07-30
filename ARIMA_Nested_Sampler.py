##Python packages
from ARIMA_fast import ARIMA_fast
import blackjax
import time
import jax.numpy as jnp
from anesthetic import NestedSamples 
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import jax

class ARIMA_Nested_Sampler:
 """
 A class to perform Nested Sampling using Blackjax Nested Sampler for ARIMA Models.
 """
 def __init__(self,data,order,log_likelihood,prior_bounds,num_live,num_delete,seed):
  """
  Initializes the Nested Sampler
  Args:
     data (array or list) : The time_series data to be fitted.
     order (tuple) : (p,d,q) order of the ARIMA model.
     lklhood (func) : The log-likelihood function containing the ARIMA model.
     prior_bounds (dict) : Bounds on the prior distribution.
     num_live (int) : number of live points to draw from the prior space
     num_delete : number of points to delete at each iteration
     seed : Seed for random number generator
      
  """
  self.data = jnp.asarray(data)
  self.order = order
  self.log_likelihood = log_likelihood
  self.prior_bounds = prior_bounds
  self.parameters = prior_bounds.keys()
  self.num_live = num_live
  self.num_delete = num_delete
  self.seed = seed
   

 def run(self):
    """
    Runs the Nested Sampling procedure for ARIMA Models.
    Args:
      num_live : number of live points to draw from the prior space.
      num_delete : number of points to delete at each iteration.
      seed : Seed for random number generator
    """
    print(f"Running Nested Sampling for fitting ARIMA {self.order} model...")
    num_dims = len(self.prior_bounds)
    num_inner_steps = num_dims * 5
    p,d,q = order
    if num_dims!=(p+q):
        raise ValueError("Number of parameters in prior_bounds inconsistent with ARIMA order.")
    
    rng_key = jax.random.PRNGKey(self.seed)
    rng_key,prior_key = jax.random.split(rng_key)
    particles,logprior_fn = blackjax.ns.utils.uniform_prior(prior_key,self.num_live,self.prior_bounds)
    ##Nested Sampler
    nested_sampler = blackjax.nss(logprior_fn=logprior_fn,loglikelihood_fn = self.log_likelihood,num_delete=self.num_delete,num_inner_steps=num_inner_steps)
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
        pbar.update(self.num_delete)
    
    dead = blackjax.ns.utils.finalise(live,dead)
    ns_time = time.time() - ns_start
    self.ns_time = ns_time
    print(f"Finished Nested Sampling with a total runtime of : {ns_time:.2f} seconds")

 
     ##Processing results
    columns = [i for i in self.prior_bounds.keys()]
    self.columns = columns
    
    data = jnp.vstack([dead.particles[key] for key in columns]).T

    posterior_samples = NestedSamples(
    data,
    logL=dead.loglikelihood,
    logL_birth=dead.loglikelihood_birth,
    columns=columns,
    labels=None,
    logzero=jnp.nan,
    )
    return posterior_samples
     #Print results:
 def summary(self,posterior_samples):
    print("||NESTED SAMPLING SUMMARY RESULTS||")
    print("---------------------------------------------------")
    print(f"Nested sampling runtime: {self.ns_time:.2f} seconds")
    print("---------------------------------------------------")
    posterior_means = []
    for key in self.prior_bounds.keys():
     means = posterior_samples[key].mean()
     print(f"Posteior mean for {key}:{means}")
     posterior_means.append(means)
     print("---------------------------------------------------")
    
    print(f"Log Evidence: {posterior_samples.logZ():.2f} ± {posterior_samples.logZ(100).std():.2f}")
    print("-------x----------------x-------------x------------")
    Z = posterior_samples.logZ()
    
    
     # Create posterior corner plot with true values marked
    kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
    axes = posterior_samples.plot_2d(self.columns, kinds=kinds, label='Posterior')
    plt.suptitle("Posterior Distributions")
    
    return posterior_means,Z
  


    


   
    



        
            
            




        
    
    
    

    


   
    



        
            
            




        
    
    
    
