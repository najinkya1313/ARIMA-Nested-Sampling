from ARIMA_fast import ARIMA_fast
import jax
import jax.numpy as jnp
import blackjax
class ARIMA_log_likelihood:
    """
    A helper class to give loglikelihood functions for some standard ARIMA models.
    Arguments:
     data : time series data for the ARIMA function.
     differencing : order of differencing (d) to be used in the ARIMA model.
    """
    def __init__(self,data,differencing):
        if differencing==None:

         self.differencing = 0
        else:
         self.differencing = differencing
        self.data = data
    
    def arima_1_d_0(self,params):
        phi_1,sigma = params['phi_1'],params['sigma']
        y_model = ARIMA_fast(self.data,(1,self.differencing,0),sigma,[phi_1],[])
        return jax.scipy.stats.multivariate_normal.logpdf(self.data,y_model,sigma**2)
    
    def arima_0_d_1(self,params):
        theta_1,sigma = params['theta_1'],params['sigma']
        y_model = ARIMA_fast(self.data,(0,self.differencing,1),sigma,[],[theta_1])
        return jax.scipy.stats.multivariate_normal.logpdf(self.data,y_model,sigma**2)
    
    def arima_1_d_1(self,params):
        phi_1,theta_1,sigma = params['phi_1'],params['theta_1'],params['sigma']
        y_model = ARIMA_fast(self.data,(1,self.differencing,1),sigma,[phi_1],[theta_1])
        return jax.scipy.stats.multivariate_normal.logpdf(self.data,y_model,sigma**2)
    
    def arima_1_d_1(self,params):
        phi_1,theta_1,sigma = params['phi_1'],params['theta_1'],params['sigma']
        y_model = ARIMA_fast(self.data,(1,self.differencing,1),sigma,[],[theta_1])
        return jax.scipy.stats.multivariate_normal.logpdf(self.data,y_model,sigma**2)
    
    def arima_0_d_0(self,params):
        if self.differencing==0:
           raise ValueError("ARIMA(0,d,0) model invalid for differencing=0!")
        sigma = params['sigma']
        y_model = ARIMA_fast(self.data,(0,self.differencing,0),sigma,[],[])
        return jax.scipy.stats.multivariate_normal.logpdf(self.data,y_model,sigma**2)
    
    def arima_2_d_0(self,params):
        phi_1,phi_2,sigma = params['phi_1'],params['phi_2'],params['sigma']
        y_model = ARIMA_fast(self.data,(1,self.differencing,1),sigma,[phi_1,phi_2],[])
        return jax.scipy.stats.multivariate_normal.logpdf(self.data,y_model,sigma**2)
    
    def arima_0_d_2(self,params):
       theta_1,theta_2,sigma = params['theta_1'],params['theta_2'],params['sigma']
       y_model = ARIMA_fast(self.data,(0,self.differencing,2),sigma,[],[theta_1,theta_2])
       return jax.scipy.stats.multivariate_normal.logpdf(self.data,y_model,sigma**2)
