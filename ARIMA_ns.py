##Python packages
import blackjax
import time
import jax.numpy as jnp
from anesthetic import NestedSamples 
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import jax
from ARIMA import ARIMA_fast
from norm_prior import normal_prior,normal_prior_unconstrained






def loglikelihood(data,order,parameters,seed):
    p,d,q = order
    def llk(params):
        sigma = params['sigma']
        parameters_modulo_sigma = list(parameters.keys())[:-1]
        arima_parameters = [params[key] for key in parameters_modulo_sigma]
        phi = arima_parameters[0:p]
        theta = arima_parameters[p:p+q]
        y_model = ARIMA_fast(data,order,0,phi,theta,seed)
        return jax.scipy.stats.multivariate_normal.logpdf(data,y_model,sigma**2)
 
    return llk

def prior_parameters(prior_type:str,order:tuple,prior_bounds={}):
    """A helper function to return the prior parameters dictionary to be used in Nested Sampling
    Arguments:
     prior_type: type of prior distribution to be used - 'normal' or 'uniform'
     order : the order (p,d,q) of ARIMA model
     prior_bounds : the bounds of prior parameters if prior_type=='uniform'
    
    """
    p,d,q = order
    prior_params = {}
    if prior_type !="uniform":
     for ar in range(p):
        prior_params.update({f'phi_{ar+1}':{'mean':0,'scale':3}})
     for ma in range(q):
        prior_params.update({f'theta_{ma+1}':{'mean':0,'scale':3}})
     
     prior_params.update({'sigma':{'mean':0,'scale':7}})
     
    elif prior_type =="uniform":
     if len(prior_bounds)==0:
        raise ValueError("Missing prior_bounds for uniform prior.")
     for ar in range(p):
        prior_params.update({f'phi_{ar+1}': prior_bounds[f'phi_{ar+1}']})
     for ma in range(q):
        prior_params.update({f'theta_{ma+1}':prior_bounds[f'theta_{ma+1}']})
     prior_params.update({'sigma':prior_bounds['sigma']})
    
    else:
       raise SyntaxError("prior_type should be normal or uniform.")

    return prior_params


class ARIMA_Nested_Sampler:
 """
 A class to perform Nested Sampling using Blackjax Nested Sampler for ARIMA Models.
 """
 def __init__(self,data,order,prior_type,num_live,num_delete,seed,prior_bounds={}):
  """
  Initializes and runs the Nested Sampling.
  Args:
     data (array or list) : The time_series data to be fitted.
     order (tuple) : (p,d,q) order of the ARIMA model.
     prior_bounds (dict) : Bounds on the prior distribution.
     num_live (int) : number of live points to draw from the prior space
     num_delete : number of points to delete at each iteration
     seed : Seed for random number generator
      
  """
  self.data = jnp.asarray(data)
  self.order = order
  
  
  self.num_live = num_live
  self.num_delete = num_delete
  self.seed = seed
  p,d,q = self.order
  
  self.prior_bounds = prior_bounds
  self.prior_type = prior_type


  prior_params = prior_parameters(self.prior_type,self.order,self.prior_bounds)
  self.prior_params = prior_params
  self.log_likelihood = loglikelihood(self.data,self.order,self.prior_params,self.seed)
 
  
    
  print(f"Running Nested Sampling for fitting ARIMA {self.order} model...")
  num_dims = len(self.prior_params)
  num_inner_steps = num_dims * 5
  p,d,q = self.order
  if num_dims!=(p+q+1):
      raise ValueError("Number of parameters in prior_bounds inconsistent with ARIMA order.")
    
  rng_key = jax.random.PRNGKey(self.seed)
  rng_key,prior_key = jax.random.split(rng_key)
  if prior_type=='normal':
   particles,logprior_fn = normal_prior(prior_key,self.num_live,self.prior_params,self.order)
  elif prior_type=='uniform':
   particles,logprior_fn = blackjax.ns.utils.uniform_prior(prior_key,self.num_live,self.prior_params)
  elif prior_type=='normal_unconstrained':
   particles,logprior_fn = normal_prior_unconstrained(prior_key,self.num_live,self.prior_params,self.order)
  else:
    raise SyntaxError(f"Invalid prior_type '{prior_type}'. prior_type should be 'normal' or 'uniform'")
  self.particles = particles
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
  columns = [i for i in self.prior_params.keys()]
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
  self.posterior_samples = posterior_samples
  Z = self.posterior_samples.logZ()
  posterior_means = []
  for key in self.prior_params.keys():
     means = self.posterior_samples[key].mean()
     posterior_means.append(means)
  self.posterior_means = posterior_means
  self.log_evidence = Z
  self.log_evidence_err = self.posterior_samples.logZ(100).std()


    

#Print results:
 def summary(self):
    print("||NESTED SAMPLING SUMMARY RESULTS||")
    print("----------------------------------------------------")
    print(f"Nested sampling runtime: {self.ns_time:.2f} seconds")
    print("----------------------------------------------------")
    for index,key in enumerate(self.prior_params.keys()):
     print(f"Posterior mean for {key} : {self.posterior_means[index]}")
    print("---------------------------------------------------")
    
    print(f"Log Evidence: {self.posterior_samples.logZ():.2f} ± {self.posterior_samples.logZ(100).std():.2f}")
    print("-------x----------------x-------------x------------")
    
    
     # Create posterior corner plot with true values marked
    kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
    axes = self.posterior_samples.plot_2d(self.columns, kinds=kinds, label='Posterior')
    plt.suptitle("Posterior Distributions")
    
    
  
 def fit_model(self,compare=None):
   y_fit = ARIMA_fast(self.data,self.order,self.posterior_means[-1],self.posterior_means[0:(self.order[0])],self.posterior_means[self.order[0]:-1],self.seed+2)
   if compare is not None:
    if compare==True:
      plt.plot(self.data,label='Data')
      plt.plot(y_fit,label=f'ARIMA {self.order} model')
    
    elif type(compare)!= bool:
      raise SyntaxError(f"Invalid value {compare} for compare argument. compare should be == True, False, or None")
    
   else:
    plt.plot(y_fit,label=f'ARIMA {self.order} model')
 
   
   
    self.y_fit = y_fit
    
    plt.legend()
    plt.xlabel('Time-step')
    plt.ylabel('Value')
    plt.show()
    return y_fit


def ARIMA_model_comparison(data,orders,prior_type,num_live,num_delete,seeds,prior_bounds={}):
    evidences = []
    for order,seed in zip(orders,seeds):
        model = ARIMA_Nested_Sampler(data,order,prior_type,num_live,num_delete,seed,prior_bounds)
        evidences.append(model.log_evidence)
    evidences = jnp.array(evidences)
    x = range(len(orders))
    max_index = jnp.where(evidences==max(evidences))[0][0]
    print(max_index)
    plt.plot(x,evidences,'o')
    plt.plot(x[max_index],evidences[max_index],marker='*',markersize=10,color='red')
    plt.axvline(x=x[max_index],c='black',linestyle='--',alpha=0.6)
    plt.xticks(ticks=range(len(orders)),labels=[str(order) for order in orders])
    plt.xlabel('ARIMA (p,d,q) order')
    plt.ylabel('Log Evidence')
    plt.title('Model Comparison Plot')
    return evidences


