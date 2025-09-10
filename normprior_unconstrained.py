import jax
import jax.numpy as jnp


def normal_prior_unconstrained(rng_key,num_live,prior_params,order):
 p,d,q = order
 prior_params_modsigma = dict(list(prior_params.items())[:-1])
 prior_params_sigma = prior_params['sigma']
 ##-------------------------------------------------Logprior function–------------------------------------------
 def logprior_fn(params):
  logprior = 0.0
  for parameter, norm_params in prior_params_modsigma.items():
   x = params[parameter]
   mean = norm_params['mean']
   scale = norm_params['scale']
   logprior += jax.scipy.stats.norm.logpdf(abs(x), mean, scale)
 
  
##For sigma:
  x_sig = params["sigma"]
  mean_sigma = prior_params_sigma['mean']
  scale_sigma = prior_params_sigma['scale']
  logprior_sigma = jax.scipy.stats.truncnorm.logpdf(x_sig,0,jnp.inf,mean_sigma,scale_sigma)
  logprior = logprior + logprior_sigma
  return logprior

##---------------------------------------------------Particle sampler-----------------------------------------------:
 @jax.jit
 def prior_sample(rng_key):
  init_keys = jax.random.split(rng_key, len(prior_params)-1)
  param_labels = [label for label in prior_params_modsigma.keys()]
  phi_labels = param_labels[0:p]
  theta_labels = param_labels[p:p+q]
  params = {}
  particles_all = jnp.array([jax.random.normal(rng_key) for rng_key in init_keys])
  phi_particles = particles_all[0:p]
  theta_particles = particles_all[p:p+q]

  rng_key,sigma_key = jax.random.split(rng_key)
  sigma_particle = 7*jax.random.truncated_normal(sigma_key,0,jnp.inf)
 
 
  for phi_label,phi_particle in zip(phi_labels,phi_particles):
      params.update({phi_label:phi_particle})
  for theta_label,theta_particle in zip(theta_labels,theta_particles):
      params.update({theta_label:theta_particle})
  params.update({'sigma':sigma_particle})
  
  

  return params

 
 particle_keys = jax.random.split(rng_key,num_live)
 particles = jax.vmap(prior_sample)(particle_keys)
 

 
   
 return particles,logprior_fn
