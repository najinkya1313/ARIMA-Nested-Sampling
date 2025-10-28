##Python packages
import blackjax
import time
import jax.numpy as jnp
from anesthetic import NestedSamples 
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import jax
from ARIMA import ARIMA_fast,ARIMA_forecast
from norm_prior import normal_prior
from fgivenx import plot_lines





def loglikelihood(data,order,seed):
    p,d,q = order
    phi_keys = [f'phi_{i+1}' for i in range(p)]
    theta_keys = [f'theta_{j+1}' for j in range(q)]
    
    def llk(params):
        sigma = params['sigma']
        mu = params['mu']
        phi = jnp.array([params[k] for k in phi_keys]) if p > 0 else jnp.array([])
        theta = jnp.array([params[k] for k in theta_keys]) if q > 0 else jnp.array([])
        if mu.shape != ():
            mu = mu.reshape(())
        
        y_model = ARIMA_fast(data,order,0,mu,phi,theta,seed)
        return jax.scipy.stats.multivariate_normal.logpdf(data,y_model,sigma**2)
 
    return llk

def prior_parameters(prior_type:str,order:tuple,scale,mu_mean,mu_scale,prior_bounds={}):
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
        prior_params.update({f'phi_{ar+1}':{'mean':0,'scale':scale}})
     for ma in range(q):
        prior_params.update({f'theta_{ma+1}':{'mean':0,'scale':scale}})
     
     prior_params.update({'sigma':{'mean':0,'scale':20}})
     prior_params.update({'mu':{'mean':mu_mean,'scale':mu_scale}})
     
    elif prior_type =="uniform":
     if len(prior_bounds)==0:
        raise ValueError("Missing prior_bounds for uniform prior.")
     for ar in range(p):
        prior_params.update({f'phi_{ar+1}': prior_bounds[f'phi_{ar+1}']})
     for ma in range(q):
        prior_params.update({f'theta_{ma+1}':prior_bounds[f'theta_{ma+1}']})
     prior_params.update({'sigma':prior_bounds['sigma']})
     prior_params.update({'mu':prior_bounds['k']})
    
    else:
       raise SyntaxError("prior_type should be normal or uniform.")

    return prior_params


class ARIMA_Nested_Sampler:
 """
 A class to perform Nested Sampling using Blackjax Nested Sampler for ARIMA Models.
 """
 def __init__(self,data,order,mu_mean,mu_scale,num_live,num_delete,seed,inner_steps_factor=6,prior_scale=1,prior_type="normal",prior_bounds={}):
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
  self.mu_mean = mu_mean
  self.mu_scale = mu_scale
  
  self.num_live = num_live
  self.num_delete = num_delete
  self.seed = seed
  self.prior_scale = prior_scale
  p,d,q = self.order
  self.prior_bounds = prior_bounds
  self.prior_type = prior_type


  prior_params = prior_parameters(self.prior_type,self.order,self.prior_scale,self.mu_mean,self.mu_scale,self.prior_bounds)
  self.prior_params = prior_params
  self.log_likelihood = loglikelihood(self.data,self.order,self.seed)
 
  
    
  print(f"Running Nested Sampling for fitting ARIMA {self.order} model...")
  num_dims = len(self.prior_params)
  num_inner_steps = num_dims * inner_steps_factor
  p,d,q = self.order
  if num_dims!=(p+q+2):
      raise ValueError("Number of parameters in prior_bounds inconsistent with ARIMA order.")
    
  rng_key = jax.random.PRNGKey(self.seed)
  rng_key,prior_key = jax.random.split(rng_key)
  if prior_type=='normal':
   particles,logprior_fn = normal_prior(prior_key,self.num_live,self.prior_params,self.order)
  elif prior_type=='uniform':
   particles,logprior_fn = blackjax.ns.utils.uniform_prior(prior_key,self.num_live,self.prior_params)

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
  labels = [fr'$\phi_{ph+1}$' for ph in range(p)] + [fr'$\theta_{th+1}$' for th in range(q)] + [r'$\sigma$',r'$\mu$']
    
  data = jnp.vstack([dead.particles[key] for key in columns]).T

  posterior_samples = NestedSamples(
  data,
  logL=dead.loglikelihood,
  logL_birth=dead.loglikelihood_birth,
  columns=columns,
  labels=labels,
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
    
 def get_mean_forecasts(self):
     p,d,q = self.order
     y_fit = ARIMA_fast(self.data,self.order,self.posterior_means[-2],self.posterior_means[-1],self.posterior_means[0:p],self.posterior_means[p:p+q],self.seed)
     return y_fit
     
  
 def fit_model(self,compare=None):
   p,d,q = self.order
   y_fit = ARIMA_fast(self.data,self.order,self.posterior_means[-2],self.posterior_means[-1],self.posterior_means[0:p],self.posterior_means[p:p+q],self.seed)
   self.y_fit = y_fit
   if compare is not None:
    if compare==True:
      plt.plot(self.data,label='Data')
      plt.plot(y_fit,label=f'ARIMA {self.order} model')
     
    elif type(compare)!= bool:
      raise SyntaxError(f"Invalid value {compare} for compare argument. compare should be == True, False, or None")
    
   else:
    plt.plot(y_fit,label=f'ARIMA {self.order} model')
 
   
   
    
    
    plt.legend()
    plt.xlabel('Time-step')
    plt.ylabel('Value')
    plt.show()

 def direct_forecast(self,overall_time,overall_data,lower_index,upper_index,num_forecast,n_samples,**kwargs):
   samples = self.posterior_samples.sample(n_samples)
   p, d, q = self.order
   samples = self.posterior_samples.sample(n_samples)

   ar_samples = [samples[f'phi_{i+1}'] for i in range(p)] if p > 0 else []
   ma_samples = [samples[f'theta_{i+1}'] for i in range(q)] if q > 0 else []
   sigma_samples = samples['sigma']
   mu_samples = samples['mu']

   posteriors = []

   for i in range(n_samples):
    ar = [ar_samples[j].iloc[i] for j in range(p)] if p > 0 else []
    ma = [ma_samples[j].iloc[i] for j in range(q)] if q > 0 else []

    sigma = sigma_samples.iloc[i]
    mu = mu_samples.iloc[i]

    posteriors.append(tuple(ar + ma + [sigma, mu]))

    
   def arima_func(x,params):
      phis = params[0:p]
      thetas = params[p:p+q]
      sigma = params[-2]
      mu = params[-1]
      y_model = ARIMA_fast(self.data,self.order,sigma,mu,phis,thetas,self.seed)
      return y_model
   
   def arima_forecast(x,params):
      phis = params[0:p]
      thetas = params[p:p+q]
      sigma = params[-2]
      mu = params[-1]
      y_forecasted = ARIMA_forecast(self.data,self.order,sigma,mu,phis,thetas,num_forecast,self.seed)
      return y_forecasted
   
   fig,axes = plt.subplots(1,1,figsize=(11,6))
   title = kwargs.get("title", "Forecast Plot")
   xlabel = kwargs.get("xlabel", "Time")
   ylabel = kwargs.get("ylabel", "Value")
   title_fontsize = kwargs.get("title_fontsize", 14)
   label_fontsize = kwargs.get("label_fontsize", 12)
   

   plot_lines(arima_func,overall_time[lower_index:upper_index],posteriors,ax=axes,color='red')
   plot_lines(arima_forecast,overall_time[upper_index:upper_index+num_forecast],posteriors,ax=axes,color='green')
   axes.plot(overall_time[lower_index:upper_index],self.data,color='black',label='Training Data')
   axes.plot(overall_time[upper_index:upper_index+num_forecast],overall_data[upper_index:upper_index+num_forecast],color='black',marker='+',linewidth=0,ms=8,label='Observed Data')
   plt.grid()
   axes.set_xlabel(xlabel,fontsize=label_fontsize)
   axes.set_ylabel(ylabel,fontsize=label_fontsize)
   plt.title(title,fontsize=title_fontsize)
   plt.legend()

    

   
   
    


def one_step_rolling(data,split_indices,num_forecast,order,seed,mu_mean=0,mu_scale=1):
    lower,upper = split_indices
    train_data = data[lower:upper]
    forecasted_points = []
    for i in range(num_forecast):
        p,d,q = order
        train_data = list(train_data)
        model = ARIMA_Nested_Sampler(train_data,order,mu_mean,mu_scale,100,50,seed)
        y_predicted = list(model.get_mean_forecasts())
        residuals = jnp.array(train_data) - jnp.array(y_predicted)
        posterior_means = model.posterior_means
        
        phi = posterior_means[0:p]
        theta = posterior_means[p:p+q]
        sigma = posterior_means[-2]
        mu = posterior_means[-1]
        if p==0:
            phi=0
        if q==0:
            theta=0
        phi_part = jnp.array(phi) * jnp.flip(jnp.array(train_data[-p:]))
        theta_part = jnp.array(theta) * jnp.flip(residuals[-q:])
        k = mu * (1- jnp.sum(jnp.array(phi)))
        error_key = jax.random.PRNGKey(seed+i)
        epsilon_t = sigma * jax.random.normal(error_key)

        y_forecast = k + jnp.sum(phi_part) + jnp.sum(theta_part) + epsilon_t
        y_predicted.append(y_forecast)
        train_data.append(data[upper+i+1])
        forecasted_points.append(y_forecast)
        print(f"Predicted value : {y_forecast}; Observed value {data[upper+i+1]}")
    return forecasted_points
        
