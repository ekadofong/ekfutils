import numpy as np
from scipy.optimize import minimize
from astropy.modeling.models import Sersic1D, Sersic2D, Const1D, Const2D
from astropy.modeling.fitting import LevMarLSQFitter
import emcee

from . import functions as ef


class LFitter ( object ):
    '''
    A lightweight Bayesian line fitter with a Gaussian likelihood including
    intrinsic scatter
    '''
    def __init__ ( self, mmin=None, mmax=None, pivot=0., bmin=None, bmax=None, smin=0.05, smax=None, nwalkers=32):
        '''
        Initialize the LFitter class with the given parameters.

        Args:
            mmin (float or None): Minimum slope of the line. Defaults to None.
            mmax (float or None): Maximum slope of the line. Defaults to None.
            pivot (float): Pivot point of the line. Defaults to 0.
            bmin (float or None): Minimum intercept of the line. Defaults to None.
            bmax (float or None): Maximum intercept of the line. Defaults to None.
            smin (float): Minimum scatter of the data points around the line. Defaults to 0.05.
            smax (float or None): Maximum scatter of the data points around the line. Defaults to None.
            nwalkers (int): Number of walkers for the MCMC sampler. Defaults to 32.
        '''        
        self.mmin = mmin if (mmin is not None) else -np.inf
        self.mmax = mmax if (mmax is not None) else np.inf
        self.bmin = bmin if (bmin is not None) else -np.inf
        self.bmax = bmax if (bmax is not None) else np.inf
        self.smin = smin
        self.smax = smax if (smax is not None) else np.inf
        self.nwalkers = nwalkers
        self.pivot = pivot
        
    def log_likelihood(self, theta, x, y, yerr, xerr=None, weights=None):
        '''
        Compute the log-likelihood of the data given the model parameters.

        Args:
            theta (array-like): Model parameters (m, b, s).
            x (array-like): Independent variable data.
            y (array-like): Dependent variable data.
            yerr (array-like): Measurement errors in the dependent variable.
            weights (array-like or None): Weights for the data points. Defaults to None.

        Returns:
            float: Log-likelihood of the data given the model parameters.
        '''
        if xerr is not None:
            x = np.random.normal(x,xerr)
            
        m, b, s = theta
        model = m * (x-self.pivot) + b
        sigma2 = yerr**2 + s**2 

        if weights is None:
            #weights = np.ones_like(y)/y.size
            wterm = 0.
        else:
            wterm = -2. * np.log(weights)
            
        # N = x.size
        # N*np.log(2.*np.pi) + 
        dev = (y - model) ** 2 / sigma2
        eterm = np.log(2.*np.pi*sigma2)
        
        return -0.5 *np.sum(  wterm + dev + eterm  )
    
    def construct_lsq_lh ( self, weights ):
        '''
        Construct the log-likelihood function to be used in a least-squares fit.

        Args:
            weights (array-like): Weights for the data points.

        Returns:
            function: Log-likelihood function to be used in a least-squares fit.
        '''        
        return lambda theta, x, y, yerr: -self.log_likelihood ( theta, x,y,yerr, weights)

    def log_prior(self, theta):
        '''
        Compute the log-prior of the model parameters.

        Args:
            theta (array-like): Model parameters (m, b, s).

        Returns:
            float: Log-prior of the model parameters.
        '''
        
        m, b, s = theta
        if (self.mmin < m < self.mmax) and (self.bmin < b < self.bmax) and (self.smin < s < self.smax):
            return 0.0
        return -np.inf

    def log_probability(self, theta, x, y, yerr, xerr, weights):
        '''
        Compute the log-probability of the model parameters given the data.

        Args:
            theta (array-like): Model parameters (m, b, s).
            x (array-like): Independent variable data.
            y (array-like): Dependent variable data.
            yerr (array-like): Measurement errors in the dependent variable.
            weights (array-like or None): Weights for the data points. Defaults to None.

        Returns:
            float: Log-probability of the model parameters given the data.
        '''        
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta, x, y, yerr, xerr, weights)
    
    def predict ( self, theta, x ):
        '''
        Predict the dependent variable values given the model parameters and independent variable data.

        Args:
            theta (array-like): Model parameters (m, b).
            x (array-like): Independent variable data.

        Returns:
            array-like: Predicted dependent variable values.
        '''        
        ypred = theta[0]*(x-self.pivot) + theta[1]
        return ypred
    
    def predict_from_run ( self, x ) :
        """
        Predicts the median and upper/lower bounds of the model for a given input value(s) using the model parameters obtained from a previous run.

        Args:
        - x: A single value or a 1D numpy array of values for which to predict the median and bounds.

        Returns:
        - prediction: A 1D numpy array with three elements representing the median and upper/lower bounds of the model predictions for the given input value(s).
        """        
        args = self.get_param_estimates ()[1]
        med = self.predict ( args[:-1], x )
        medup = self.predict ( [args[0], args[1]+args[2], args[2]], x )
        meddown = self.predict ( [args[0], args[1]-args[2], args[2]], x )
        prediction = np.array([med,meddown, medup])
        return prediction
    
    def rms ( self, theta, x, y ):
        """
        Calculates the root-mean-square error between the model predictions and actual values.

        Args:
        - theta: A 1D numpy array representing the model parameters.
        - x: A 1D numpy array of input values.
        - y: A 1D numpy array of actual output values corresponding to the input values.

        Returns:
        - rmse: A float representing the root-mean-square error between the model predictions and actual values.
        """        
        ypred = theta[0]*x + theta[1] 
        return np.sum(( y - ypred )**2)
    
    def do_lsquares ( self, x, y, yerr, initial, weights=None):
        '''
        Perform a least squares minimization from an initial guess.

        Parameters:
        -----------
        x : array-like
            The x-values of the data.
        y : array-like
            The y-values of the data.
        yerr : array-like
            The uncertainties of the y-values of the data.
        initial : array-like
            The initial guess for the parameters to be fitted.
        weights : array-like or None, optional
            The weights to be used for the fit. If None, all data points will
            have equal weight.

        Returns:
        --------
        x_fit : array-like
            The fitted parameters that minimize the sum of squares of the residuals.

        Notes:
        ------
        The least squares minimization is performed using the scipy.optimize.minimize 
        function with the L-BFGS-B method. bounds on the parameters can be set with the 
        bounds argument.
        '''
        nll = self.construct_lsq_lh(weights=weights)
        soln = minimize(nll, initial, args=(x, y, yerr), bounds=((self.mmin,self.mmax),
                                                                 (self.bmin,self.bmax),
                                                                 (self.smin,self.smax)))
        self.lsquares = soln 
        return soln.x

    def run (self, x,y,yerr, initial, xerr=None, weights=None, steps=500, seed=None, progress=True):
        '''
        Run the MCMC inference
        '''    
        pout = np.array(initial) 
        if seed is None:
            rs = np.random
        else:
            ra = np.random.RandomState ( seed )
        
        if pout.ndim == 1:    
            pos = rs.normal ( pout, abs(pout+1e-3)*.25, [self.nwalkers,len(initial)] )
            
        else:
            pos = pout
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability, args=(x, y, yerr, xerr,  weights)
        )
        sampler.run_mcmc(pos, steps, progress=progress)
        self.sampler = sampler

    def get_param_estimates ( self, fdiscard=0.6 ):
        """
        Calculates the median and 16th/84th percentiles of the posterior distribution of the model parameters.
        """        
        if hasattr ( self, 'parameter_estimates'):
            return self.parameter_estimates
        discard = int(self.sampler.get_chain().shape[0]*fdiscard)
        fchain = self.sampler.get_chain(flat=True, discard=discard)
        return np.quantile(fchain, [0.16,.5,.84], axis=0)
    
    def set_param_estimates ( self, param_array ):
        self.parameter_estimates = param_array
        
    def get_uncertainties (self, ms, alpha=0.32, npull=10000, discard=100):
        """
        Get upper and lower bound estimates [16th/84th] in the prediction space
        """
        #ax.plot ( ms, ms*pout[0] + pout[1], color=colors_d[source].modulate(0.4,0.).base, zorder=0 )
        fchain = self.sampler.get_chain(flat=True, discard=discard)
        parr = np.zeros([npull, ms.size])
        for idx in np.arange(parr.shape[0]):
            pull = fchain[np.random.randint(0, fchain.shape[0])]
            parr[idx] = self.predict ( pull, ms ) #(ms-self.pivot)*pull[0] + pull[1]
        return np.quantile(parr, [alpha/2.,.5,1.-alpha/2.], axis=0)
    
    def plot_uncertainties (self, ms, ax, color='k', erralpha=0.1, alpha=0.32, discard=100, yscale='linear', label=None, show_std=True, lw=2, **kwargs ):
        predictions = self.get_uncertainties ( ms, alpha=alpha, discard=discard  )
        params = self.get_param_estimates ()
        
        if yscale=='linear':
            ax.plot(ms, predictions[1],color=color, label=label, **kwargs)
            ax.fill_between(ms, predictions[0], predictions[2], alpha=erralpha, color=color, **kwargs)
            if show_std:                
                for sign in [-1.,1.]:
                    ax.plot(ms, predictions[1] + sign*params[1,2], color=color, label=label, **kwargs)
        elif yscale=='log':
            ax.plot(ms, 10.**predictions[1],color=color, label=label, lw=lw, **kwargs)
            ax.fill_between(ms, 10.**predictions[0], 10.**predictions[2], alpha=erralpha, color=color, **kwargs)    
            if show_std:
                for sign in [-1.,1.]:
                    ax.plot(ms,10.**(predictions[1] + sign*params[1,2]), color=color, label=label, lw=lw/2., **kwargs)                    
        return ax
    
    def plot_chain (self, fsize=2):
        import matplotlib.pyplot as plt 
        
        chain = self.sampler.get_chain ()
        
        fig, axarr = plt.subplots(self.sampler.ndim, 1, figsize=(10,fsize*self.sampler.ndim))
        for aindex, ax in enumerate(axarr):
            for windex in range(self.sampler.nwalkers):
                ax.plot ( chain[:, windex, aindex], color='lightgrey', alpha=0.3)    
    
    
class FlexFitter ( LFitter ):
    def athirteen_mzr ( self, theta, logmstar ):
        zp, logmto, gamma = theta
        massratio = 10.**(logmto - logmstar)        
        return zp - np.log10(1. + massratio**gamma)
    
    def gaussian ( self, theta, x ):
        A, m, s, c = theta
        return ef.gaussian(x,A,m,s) + c
    
    def set_predictor ( self, functional_form='a13', prior_args=None ):
        if functional_form == 'a13':
            self.predict = self.athirteen_mzr            
        elif functional_form == 'gaussian':
            self.predict = self.gaussian
        self.prior_args = prior_args

    def uniform_log_prior(self, theta):
        if self.prior_args is None:
            return 0.
        else:
            for idx, param in enumerate(theta):                
                if (param > self.prior_args[idx][1]) or (param < self.prior_args[idx][0]):                    
                    return -np.inf
            return 0.
    
    def gaussian_log_prior ( self, theta ):
        lnp = 0
        for idx, param in enumerate(theta):
            probparam = ef.gaussian( param,'normalize', self.prior_args[idx][0], self.prior_args[idx][1] )
            lnp += np.log(probparam)
            #if (param > self.prior_args[idx][1]) or (param < self.prior_args[idx][0]):
            #    return -np.inf
        return lnp     
            
    def set_prior ( self, form='uniform' ):
        if form == 'uniform':
            self.log_prior = self.uniform_log_prior
            self.bounds = self.prior_args
        elif form == 'gaussian':
            self.log_prior = self.gaussian_log_prior
            self.bounds = [ (x[0]*.5, x[0]*1.5) for x in self.prior_args ]
                    
    def do_lsquares ( self, x, y, yerr, initial, weights=None):
        nll = self.construct_lsq_lh(weights=weights)        
        soln = minimize(nll, initial, args=(x, y, yerr), bounds=self.bounds)
        self.lsquares = soln 
        return soln.x     
    
    def log_likelihood(self, theta, x, y, yerr, xerr=None, weights=None):
        '''
        Gaussian logL
        '''        
        if xerr is not None:
            x = np.random.normal(x,xerr)
        
        model = self.predict ( theta[:-1], x ) 
        sigma2 = yerr**2 + theta[-1]**2 
        
        if weights is None:
            weights = 1. #np.zeros_like(y)#/y.size
        
        N = x.size
        dev = (y - model) ** 2 / sigma2
        eterm = np.log(2.*np.pi*sigma2)
        #wterm = -2. * np.log(weights)
        return -0.5 * np.sum(  weights * (dev + eterm)  )
    

def fit_sersic_1d(radius, intensity, init_n=1., init_r_eff=None, init_const=0., fixed_parameters=None):
    if init_r_eff is None:        
        #r = np.sqrt ( (x-init_x_0)**2 + (y-init_y_0)**2 )
        init_r_eff = np.trapz(radius*intensity*2.*np.pi*radius, radius)
        init_r_eff /=    np.trapz(radius*2.*np.pi*radius, radius)   

    init_amplitude = np.nanmax(intensity)
    sersic_init = Sersic1D(
        amplitude=init_amplitude, 
        r_eff=init_r_eff, 
        n=init_n, 
    )
    const_init = Const1D(
        amplitude=init_const
    )
    sersic_init = sersic_init + const_init
    
    if fixed_parameters is not None:
        for param in fixed_parameters:            
            setattr(getattr(sersic_init, param), 'fixed', True)
    
    sersic_init.bounds.update ({
        'amplitude_0': (init_amplitude*0.1, np.inf),
        'r_eff_0': (0, radius.max()),
        'n_0': (0.5, 10),  # Sersic index typically ranges from 0.1 to 10
        'amplitude_1': (0., np.inf)
    })
    fitter = LevMarLSQFitter()

    fitted_model = fitter(sersic_init, radius, intensity)
    im = fitted_model(radius)
    return fitted_model, im              

def fit_sersic_2d(image, init_n=1., init_r_eff=None, init_ellip=0.5, init_theta=0., init_x_0=None, init_y_0=None, fixed_parameters=None, nan_replace=0.):
    image = np.where(np.isnan(image), nan_replace, image)
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    
    if init_x_0 is None:
        init_x_0 = int(image.shape[1] / 2)
    if init_y_0 is None:
        init_y_0 = int(image.shape[0] / 2)
    init_amplitude = image[int(init_y_0),int(init_x_0)]
    
    if init_r_eff is None:        
        #r = np.sqrt ( (x-init_x_0)**2 + (y-init_y_0)**2 )
        init_r_eff = min(image.shape) / 10    
        print(init_r_eff)    

    sersic_init = Sersic2D(
        amplitude=init_amplitude, 
        x_0=init_x_0, 
        y_0=init_y_0, 
        r_eff=init_r_eff, 
        n=init_n, 
        ellip=init_ellip, 
        theta=init_theta
    )
    if fixed_parameters is not None:
        for param in fixed_parameters:
            #sersic_init.x_0.fixed = True
            setattr(getattr(sersic_init, param), 'fixed', True)
                
    sersic_init.bounds.update ({
        'amplitude': (init_amplitude*0.1, np.inf),
        'x_0': (0, image.shape[1]),
        'y_0': (0, image.shape[0]),
        'r_eff': (0, max(image.shape) / 2),
        'n': (0.1, 10)  # Sersic index typically ranges from 0.1 to 10
    })
    fitter = LevMarLSQFitter()

    fitted_model = fitter(sersic_init, x, y, image)
    im = fitted_model(x,y)
    return fitted_model, im

class BaseInferer (object):
    def set_uniformprior ( self, bounds ):
        self.ndim = len(bounds)
        self.bounds = bounds
        def logprior ( theta ):
            for pidx,parameter in enumerate(theta):
                if parameter < bounds[pidx][0]:
                    return -np.inf
                elif parameter > bounds[pidx][1]:
                    return -np.inf
            return 0.
        self.logprior = logprior    
        
    def set_bounds ( self, bounds ):   
        if hasattr ( self, 'ndim' ):
            assert len(bounds) == self.ndim
        else:
            self.ndim = len(bounds)
        self.bounds = bounds    
    
    def set_loglikelihood ( self, likelihood_fn ):
        self.loglikelihood = likelihood_fn
        
    def set_logprior ( self, logprior_fn ):
        self.logprior = logprior_fn
    
    def logprob ( self, theta, data ):
        lnP = 0.
        lnP += self.logprior ( theta )
        if not np.isfinite(lnP):
            return -np.inf
        lnP += self.loglikelihood ( theta, data )
        return lnP    
    
    def run (self, data, initial=None, steps=500, nwalkers=32, progress=True):
        '''
        Run the MCMC inference
        ''' 
        if initial is None:
            if hasattr ( self, 'bounds' ):
                initial = [ np.random.uniform(self.bounds[idx][0],self.bounds[idx][1], nwalkers) for idx in range(len(self.bounds))]
                initial = np.array(initial).T
        
        if not hasattr(self,'ndim'):
            self.ndim = np.shape(initial)[1]
        
        sampler = emcee.EnsembleSampler(
            nwalkers, self.ndim, self.logprob, args=(data,)
        )
        sampler.run_mcmc(initial, steps, progress=progress)
        self.sampler = sampler   
        
    def get_param_estimates ( self, fdiscard=0.6, alpha=0.32 ):
        """
        Calculates the median and 16th/84th percentiles of the posterior distribution of the model parameters.
        """        
        if hasattr ( self, 'parameter_estimates'):
            return self.parameter_estimates
        discard = int(self.sampler.get_chain().shape[0]*fdiscard)
        fchain = self.sampler.get_chain(flat=True, discard=discard)
        return np.quantile(fchain, [alpha/2.,.5,1. - alpha/2.], axis=0)            
        
    def plot_chain (self, fsize=2):
        import matplotlib.pyplot as plt 
        
        chain = self.sampler.get_chain ()
        
        fig, axarr = plt.subplots(self.sampler.ndim, 1, figsize=(10,fsize*self.sampler.ndim))
        if self.ndim == 1:
            axarr = [axarr]
        for aindex, ax in enumerate(axarr):
            for windex in range(self.sampler.nwalkers):
                ax.plot ( chain[:, windex, aindex], color='lightgrey', alpha=0.3)     
                
        return fig, axarr        

def plawparams_from_pts (xs, ys):
    m = (ys[1]-ys[0])/(xs[1]-xs[0])
    b = ys[0] - xs[0]*m
    return m,b

def ellipse_from_point ( x,y, ellip=0., theta=0. ):
    '''
    Get the semimajor axis of an ellipse with fixed PA (theta) and
    ellipticity (ellip) given that it is sampled at (x,y)
    '''
    phi = np.arctan2(y,x) - theta # angle between vector along semimajor axis and sampled point
    hypot = np.sqrt(x**2 + y**2) # distance from origin
    
    semimajor = hypot * np.sqrt ( np.cos(phi)**2 + np.sin(phi)**2/(1.-ellip)**2 )
    return semimajor

