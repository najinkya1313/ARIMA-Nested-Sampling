import jax
import jax.numpy as jnp


def normal_prior(rng_key,num_live,prior_params,order):
 p,d,q = order
 prior_params_modsigma = dict(list(prior_params.items())[:-1])
 prior_params_sigma = prior_params['sigma']
 ##-------------------------------------------------Logprior function–------------------------------------------
 def logprior_fn(params):
  logprior = 0.0
  parameters_mod_sigma = jnp.array([-params[key] for key in prior_params_modsigma.keys()])
  ar_parameters = jnp.flip(parameters_mod_sigma[0:p])
  ma_parameters = jnp.flip(parameters_mod_sigma[p:p+q])
  const = jnp.ones(1)
  poly_ar = jnp.concatenate([ar_parameters,const])
  poly_ma = jnp.concatenate([ma_parameters,const])
  roots_phi = jnp.roots(poly_ar,strip_zeros=False)
  roots_ma = jnp.roots(poly_ma,strip_zeros=False)
 
  for parameter, norm_params in prior_params_modsigma.items():
   x = params[parameter]
   mean = norm_params['mean']
   scale = norm_params['scale']
   logprior += jax.scipy.stats.norm.logpdf(abs(x), mean, scale)
  output_phi = jnp.where(jnp.all(abs(roots_phi)>1) ,logprior,-jnp.inf)
  output_ma = jnp.where(jnp.all(abs(roots_ma)>1),0,-jnp.inf)
  output = output_phi + output_ma
  
##For sigma:
  x_sig = params["sigma"]
  mean_sigma = prior_params_sigma['mean']
  scale_sigma = prior_params_sigma['scale']
  logprior_sigma = jax.scipy.stats.truncnorm.logpdf(x_sig,1e-5,10,mean_sigma,scale_sigma)
  logprior = output + logprior_sigma
  return logprior

##---------------------------------------------------Particle sampler-----------------------------------------------:
 def prior_sample(rng_key):
  init_keys = jax.random.split(rng_key, len(prior_params)-1)
  param_labels = [label for label in prior_params_modsigma.keys()]
  phi_labels = param_labels[0:p]
  theta_labels = param_labels[p:p+q]
  params = {}
  particles_all = jnp.array([jax.random.normal(rng_key) for rng_key in init_keys])
 

  rng_key,sigma_key = jax.random.split(rng_key)
  sigma_particle = jax.random.truncated_normal(sigma_key,0,5)
 
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
    return params
  def invalid_point(roots):
    for phi_label,phi_particle in zip(phi_labels,phi_particles):
      params.update({phi_label:0.})
    for theta_label,theta_particle in zip(theta_labels,theta_particles):
      params.update({theta_label:0.})
    params.update({'sigma':0.})
    return params
  filtered_params = jax.lax.cond(jnp.all(abs(roots)>1),valid_point,invalid_point,roots)

  return filtered_params
 ##--------------------------------------------------------------------------------------------------------

 
 ##--------------------------------------Filter to only accept valid particles-------------------------------
 def particles_filter(particles):
   init_logprior = jax.vmap(logprior_fn)(particles)
   mask = jnp.where(init_logprior!=-jnp.inf)
   valid_particles = {}
   for key,vals in particles.items():
     valid_particles.update({key:vals[mask]})
   valid_particles = {label:value[:num_live] for label,value in valid_particles.items()}
   return valid_particles
 
 particle_keys = jax.random.split(rng_key,num_live*100)
 unfiltered_particles = jax.vmap(prior_sample)(particle_keys)
 particles = particles_filter(unfiltered_particles)

 ##------------------While loop to keep drawing samples until num_live reached (optimize with jax later)--------------------------------
 while len(particles['sigma'])<num_live:
   rng_key,sample_key = jax.random.split(rng_key)
   sample_particle_keys = jax.random.split(sample_key,num_live)
   new_particles = jax.vmap(prior_sample)(sample_particle_keys)
   new_particles_filtered = particles_filter(new_particles)
   for key,vals in new_particles_filtered.items():
     new_arr = jnp.concatenate([particles[key],vals])
     particles.update({key:new_arr})
  
 particles = {label:value[:num_live] for label,value in particles.items()}
   
 return particles,logprior_fn
