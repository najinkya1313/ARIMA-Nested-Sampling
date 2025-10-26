import jax
import jax.numpy as jnp


def normal_prior(rng_key,num_live,prior_params,order):
 p,d,q = order
 phi_names = [f"phi_{i+1}" for i in range(p)]
 theta_names = [f"theta_{j+1}" for j in range(q)]
 prior_params_modsigma = {name: prior_params[name] for name in phi_names + theta_names}
 prior_params_sigma = prior_params['sigma']
 mean_sigma = prior_params_sigma['mean']
 scale_sigma = prior_params_sigma['scale']
 prior_params_mu = prior_params['mu']
 mu_mean = prior_params_mu['mean']
 mu_scale = prior_params_mu['scale']
 overall_scale = list(prior_params_modsigma.values())[0]['scale']
 
 ##-------------------------------------------------Logprior function–------------------------------------------
 def logprior_fn(params):
  logprior = 0.0
  parameters_mod_sigma = jnp.array([params[key] for key in prior_params_modsigma.keys()])
  ar_parameters = jnp.flip(parameters_mod_sigma[0:p])
  ma_parameters = jnp.flip(parameters_mod_sigma[p:p+q])
  const = jnp.ones(1)
  poly_ar = jnp.concatenate([-ar_parameters,const])
  poly_ma = jnp.concatenate([ma_parameters,const])
  roots_phi = jnp.roots(poly_ar,strip_zeros=False)
  roots_ma = jnp.roots(poly_ma,strip_zeros=False)
 
  for parameter, norm_params in prior_params_modsigma.items():
   x = params[parameter]
   mean = norm_params['mean']
   scale = norm_params['scale']
   logprior += jax.scipy.stats.norm.logpdf(x, mean, scale)
  output_phi = jnp.where(jnp.all(abs(roots_phi)>1) ,logprior,-jnp.inf)
  output_ma = jnp.where(jnp.all(abs(roots_ma)>1),logprior,-jnp.inf)
  output = output_phi + output_ma
  
##For sigma and k:
  x_sig = params["sigma"]
  mu = params['mu']
  logprior_sigma = jax.scipy.stats.truncnorm.logpdf(x_sig,1e-5,jnp.inf,mean_sigma,scale_sigma)
  logprior_mu = jax.scipy.stats.norm.logpdf(mu,mu_mean,mu_scale)
  logprior = output + logprior_sigma + logprior_mu
  return logprior

##---------------------------------------------------Particle sampler-----------------------------------------------:
 @jax.jit
 def prior_sample(rng_key):
  
  init_keys = jax.random.split(rng_key, len(prior_params)-2)
  param_labels = [label for label in prior_params_modsigma.keys()]
  phi_labels = param_labels[0:p]
  theta_labels = param_labels[p:p+q]
  params = {}
  particles_all = overall_scale*jnp.array([jax.random.normal(rng_key) for rng_key in init_keys])
 

  rng_key,sigma_key = jax.random.split(rng_key)
  sigma_particle = scale_sigma*(jax.random.truncated_normal(sigma_key,1e-5,jnp.inf)) + mean_sigma

  rng_key,mu_key = jax.random.split(rng_key)
  mu_particle = mu_mean + mu_scale*(jax.random.normal(mu_key))
 
 #Roots calculation
  phi_particles = particles_all[0:p]
  theta_particles = particles_all[p:p+q]
  theta_particles_flipped = jnp.flip(theta_particles)
  phi_particles_flipped = jnp.flip(phi_particles)
  const = jnp.ones(1)
  phi_poly = jnp.concatenate([-phi_particles_flipped,const])
  theta_poly = jnp.concatenate([theta_particles_flipped,const])
  roots_phi = jnp.roots(phi_poly,strip_zeros=False)
  roots_theta = jnp.roots(theta_poly,strip_zeros=False)
  roots = jnp.concatenate([roots_phi,roots_theta])

  def valid_point(roots):
    for phi_label,phi_particle in zip(phi_labels,phi_particles):
      params.update({phi_label:phi_particle})
    for theta_label,theta_particle in zip(theta_labels,theta_particles):
      params.update({theta_label:theta_particle})
    params.update({'sigma':sigma_particle})
    params.update({'mu':mu_particle})
    return params
  def invalid_point(roots):
    for phi_label,phi_particle in zip(phi_labels,phi_particles):
      params.update({phi_label:0.})
    for theta_label,theta_particle in zip(theta_labels,theta_particles):
      params.update({theta_label:0.})
    params.update({'sigma':0.})
    params.update({'mu':0.})
    return params
  
  filtered_params = jax.lax.cond(jnp.all(abs(roots)>1),valid_point,invalid_point,roots)
  initlogprior = logprior_fn(filtered_params)
  
  return filtered_params,initlogprior
 ##--------------------------------------------------------------------------------------------------------

 
 ##--------------------------------------Filter to only accept valid particles-------------------------------
 def particles_filter(unfiltered_particles,initlogprior):
   
   
   mask = initlogprior != -jnp.inf
   
   for key,vals in unfiltered_particles.items():
     unfiltered_particles.update({key:vals[mask]})
   
   return unfiltered_particles
 
 particle_keys = jax.random.split(rng_key,num_live*1000)
 unfiltered_particles,unfilteredlogprior = jax.vmap(prior_sample)(particle_keys)
 particles = particles_filter(unfiltered_particles,unfilteredlogprior)
 
 ##------------------While loop to keep drawing samples until num_live reached --------------------------------
 while len(particles['sigma'])<num_live:
   rng_key,sample_key = jax.random.split(rng_key)
   sample_particle_keys = jax.random.split(sample_key,num_live*1000)
   new_particles,newlogprior = jax.vmap(prior_sample)(sample_particle_keys)
   new_particles_filtered = particles_filter(new_particles,newlogprior)
   for key,vals in new_particles_filtered.items():
     new_arr = jnp.concatenate([particles[key],vals])
     particles.update({key:new_arr})
   
 particles = {label:value[:num_live] for label,value in particles.items()}
   
 return particles,logprior_fn
