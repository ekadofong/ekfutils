import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.integrate import fixed_quad
from scipy.optimize import minimize, curve_fit
from astropy.modeling.models import Sersic1D, Sersic2D, Const1D, Const2D
from astropy.modeling.fitting import LevMarLSQFitter
import emcee

from ekfplot import colors as ec
from ekfplot import legend

from . import functions as ef
from . import sampling 



   
    

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
    def __init__(self) -> None:
        self.has_intrinsic_dispersion = False
        
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
        
    def chain_convergence_statistics (self, fdiscard=0.):
        chain = self.sampler.get_chain ()
        ndiscard = int(chain.shape[0]*fdiscard)
        chain = chain[ndiscard:]       
        gr_stats = sampling.gelmanrubin(chain)
        return gr_stats
    
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
    
    
    def run (self, data, initial=None, steps=500, nwalkers=32, progress=True, niter=10):
        '''
        Run the MCMC inference
        ''' 
        if initial is None:
            if hasattr ( self, 'bounds' ):
                initial = [ np.random.uniform(self.bounds[idx][0],self.bounds[idx][1], 10*nwalkers) for idx in range(len(self.bounds))]
                initial = np.array(initial).T
                is_valid = np.array([ np.isfinite(self.logprob(initial[idx], data )) for idx in range(nwalkers*10) ])
                if not is_valid.sum() >= nwalkers:
                    raise ValueError ("Initial boundaries did not produce finite-valued initial walker positions!")
            
                initial = initial[is_valid][:nwalkers]
            else:
                raise ValueError("No initial walker positions or bounds provided!")
        
        if not hasattr(self,'ndim'):
            self.ndim = np.shape(initial)[1]
        
        sampler = emcee.EnsembleSampler(
            nwalkers, self.ndim, self.logprob, args=(data,)
        )

        sampler.run_mcmc(initial, steps, progress=progress)
        
        idx = 1
        while idx < niter:
            # \\ based off of DFM recommendation here:
            # \\ https://groups.google.com/g/emcee-users/c/fg7sQNw8YcU?pli=1
            # \\ My usual recommendation here is to run a burn in as follows:
            # \\ 
            # \\ 1. Run a short (few hundred steps) chain
            # \\ 2. Reinitialize all the walkers near the point with maximum log probability seen so far
            # \\ 3. Return to step 1 a few times
            # \\ 4. Then run your final chain starting where you ended up for your last run of step 1
            # \\ 
            # \\ That normally does the trick! 
            # \\
            # \\  Gelman and Rubin (1992) and Brooks and Gelman (1998) suggest that 
            # \\ diagnostic Rc values greater than 1.2 for any of the model parameters should indicate nonconvergence
            if progress:
                print('Re-initializing walkers...')
            fullchains = sampler.get_chain()
            chains = fullchains[(idx-1)*steps:]
            gr_statistic = sampling.gelmanrubin(fullchains)
            if progress:
                print(f'max(GR) = {max(gr_statistic):.3f}')
            if (gr_statistic < 1.2).all():
                print('Convergence achieved')
                idx = niter + 1
            else:                
                
                lp_est = np.median(chains,axis=(0,1))
                std_chains = np.subtract(*np.nanquantile(chains,[0.6,.4], axis=(0,1),)) # \\ restrictive)
                
                new_initial_positions = np.random.normal(lp_est, std_chains, [nwalkers, sampler.ndim])
                sampler.run_mcmc(new_initial_positions, steps, progress=progress)
                idx += 1
        
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
    
    def predict_from_run ( self, x, stochastic=False ) :
        """
        Predicts the MaPost model for a given input value(s) using the model parameters obtained from a previous run.

        Args:
        - x: A single value or a 1D numpy array of values for which to predict the median and bounds.

        Returns:
        - prediction: A 1D numpy array with three elements representing the median and upper/lower bounds of the model predictions for the given input value(s).
        """        
        assert hasattr(self, 'predict'), "No prediction function stored!"
        if not stochastic:
            args = self.get_param_estimates ()[1]
        else:
            if not hasattr(self, 'fchain'):
                fchain = self.sampler.get_chain(flat=True,)
                self.fchain = fchain
            else:
                fchain = self.fchain
            
            args = fchain[np.random.randint(fchain.shape[0])]
            
        if self.has_intrinsic_dispersion:
            args = args[:-1]
        prediction = self.predict ( x, *args )
        return prediction    
        
    def plot_chain (self, labels=None, fsize=2, truth=None, discard=0):
        import matplotlib.pyplot as plt 

        chain = self.sampler.get_chain (discard=discard)
        
        fig, axarr = plt.subplots(self.sampler.ndim, 1, figsize=(10,fsize*self.sampler.ndim))
        if self.ndim == 1:
            axarr = [axarr]
        for aindex, ax in enumerate(axarr):
            for windex in range(self.sampler.nwalkers):
                ax.plot ( chain[:, windex, aindex], color='grey', alpha=0.1)     
            ax.axhline ( np.median(chain[:,:,aindex]), color='tab:blue', lw=2)
            for qt in [0.16,.84]:
                ax.axhline ( np.quantile(chain[:,:,aindex], qt), color='tab:blue', ls='--', lw=0.5)
            ax.axhspan ( *np.quantile(chain[:,:,aindex], [0.16,.84]),  color='tab:blue', alpha=0.2)
            if labels is not None:
                ax.set_ylabel(labels[aindex])
            if truth is not None:
                ax.axhline(truth[aindex], lw=2, color='r')
            
                
        return fig, axarr    
    
    def set_predict ( self, model_fn ):
        '''
        Set predict function, of form:
        self.predict ( data, *parameters )
        '''
        self.predict = model_fn   
        
        
    def define_poissonpointprocess_likelihood ( self, rate_fn=None, domain=[-np.inf, np.inf], volume=1. ):
        rate_fn = self.predict
        def lnP(theta, data):
            '''
            Compute the log-likelihood of a weighted Poisson point process
            
            lnL = sum_i^N ( w_i * ln(f(x_i)) ) - \int f(x) dx
            '''
            x_values,weights = data
            total_number = volume*fixed_quad(lambda x: rate_fn(x,*theta), *domain)[0] # total number or number density            
            
            expected_rates = weights*np.log(rate_fn(x_values, *theta))         

            log_likelihood = np.sum(expected_rates) - total_number
            return log_likelihood
        return lnP 
    
    def define_gaussianlikelihood (self, model_fn, with_intrinsic_dispersion=True, remove_nan=True ):
        if with_intrinsic_dispersion:
            self.has_intrinsic_dispersion = True
        else:
            self.has_intrinsic_dispersion = False     
               
        def lnP ( theta, data ):
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
            x, y, yerr, xerr = data
            if with_intrinsic_dispersion:
                intr_s = theta[-1]
                theta = theta[:-1]                
            else:
                intr_s = 0.                 
            if xerr is None:
                xerr = 0.
            
            
            model = model_fn(x,*theta)
            sigma2 = yerr**2 + intr_s**2 + xerr**2 # \\ XXX Check that this is right!!

            dev = (y - model) ** 2 / sigma2
            
            eterm = np.log(2.*np.pi*sigma2)
            if remove_nan:
                return -0.5 * np.nansum(  dev + eterm  ) 
            else:
                return -0.5 * np.sum(  dev + eterm  ) 
        
        return lnP
    
    def estimate_y (self, ms, alpha=0.32, npull=10000, discard=100):
        """
        Get upper and lower bound estimates [16th/84th] in the prediction space
        """
        if isinstance(ms, float):
            ms = np.array(ms)
        #ax.plot ( ms, ms*pout[0] + pout[1], color=colors_d[source].modulate(0.4,0.).base, zorder=0 )
        fchain = self.sampler.get_chain(flat=True, discard=discard)
        parr = np.zeros([npull, ms.size])
        for idx in np.arange(parr.shape[0]):
            pull = fchain[np.random.randint(0, fchain.shape[0])]
            if self.has_intrinsic_dispersion:
                pull = pull[:-1]
            parr[idx] = self.predict ( ms, *pull )
        if alpha is None:
            return parr
        else:
            return np.quantile(parr, [alpha/2.,.5,1.-alpha/2.], axis=0)  
    
    @property
    def mapo (self):
        return self.sampler.get_chain(flat=True)[np.argmax(self.sampler.get_log_prob(flat=True))]        
    
    def plot_uncertainties (self, ms, ax=None, color='k', transparency=0.9, alpha=0.32, 
                            discard=100,xscale='linear', yscale='linear', label=None, show_std=True, lw=2, show_median=True, **kwargs ):
        from ekfplot.plot import outlined_plot
        
        if ax is None:
            ax = plt.subplot(111)
        
        predictions = self.estimate_y ( ms, alpha=alpha, discard=discard  )
        if xscale == 'linear':
            plot_ms = ms
        elif xscale == 'log':
            plot_ms = 10.**ms
        if yscale=='linear':
            if show_median:
                outlined_plot(plot_ms, predictions[1],color=color, label=label, lw=lw, ax=ax, **kwargs)
            ax.fill_between(plot_ms, predictions[0], predictions[2], alpha=1.-transparency, color=color, **kwargs)
        elif yscale=='log':
            if show_median:
                outlined_plot(plot_ms, 10.**predictions[1],color=color, label=label, lw=lw, ax=ax, **kwargs)
            ax.fill_between(plot_ms, 10.**predictions[0], 10.**predictions[2], alpha=1.-transparency, color=color, **kwargs)    
        
        if self.has_intrinsic_dispersion and show_std:
            intdisp = self.get_param_estimates ()[1,-1]
            for sign in [-1.,1.]:
                outlined_plot(plot_ms, predictions[1] + sign*intdisp,color=color, lw=lw*.85, ax=ax, **kwargs)
                
        
        return ax    
    
    def help ( self ):
        print('''
bi = fit.BaseInferer()
bi.set_predict ( lambda x, m, b: m*x + b  )
lnP = bi.define_gaussianlikelihood( bi.predict, with_intrinsic_dispersion=True )
bi.set_loglikelihood(lnP)
bi.set_uniformprior([[-10.,0.],[10.,40.], [0.,10.]])      

data = (x,y,yerr,xerr=None)
              ''')

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

def poly2d ( x, y, coeffs, deg ):
    z_pred = np.zeros_like(x)
    tuples_list = [(i, j) for i in range(deg+1) for j in range(deg+1) if i + j <= deg]    
    
    expected_length = (deg+1)*(deg+2)//2
    assert len(tuples_list) == expected_length
    for idx,(i,j) in enumerate(tuples_list):
        #k = i+j        
        z_pred += coeffs[idx] * x**i * y**j
    return z_pred

def print_poly2d ( coeffs, deg ):
    tuples_list = [(i, j) for i in range(deg+1) for j in range(deg+1) if i + j <= deg]    
    st = []
    for idx,(i,j) in enumerate(tuples_list):
        if (i==0) and (j>0):
            term = f'{coeffs[idx]:.4f}y^{j}'
        elif (j==0) and (i>0):
            term = f'{coeffs[idx]:.4f}x^{i}'
        elif (i==0) and (j==0):
            term = f'{coeffs[idx]:.4f}'
        else:
            term = f'{coeffs[idx]:.4f}x^{i}y^{j}'
        st.append(term)
    st = " + ".join(st)
    print(st)

def polyfit2d ( x, y, z, deg):
    x,y,z = sampling.fmasker(x,y,z)
    xy = np.vstack((x,y))
    
    ncoeffs = (deg + 1)*(deg + 2)//2
    poly_spec = lambda xy, *args: poly2d ( xy[0], xy[1], args, deg)
    
    initial_guess = (np.ones(ncoeffs, dtype=float), )
    params, covariance = curve_fit(poly_spec, xy, z, p0=initial_guess)
    
    return params, covariance

def crosscorrelate(ts, x, y, return_tau=True, normalize=True):
    """
    Computes the cross-correlation between two signals, optionally returning the lag at which the
    maximum correlation occurs and normalizing the correlation coefficients.

    Parameters:
        ts (array_like): Time stamps for the signal data points. Assumed to be equally spaced.
        x (array_like): First signal array.
        y (array_like): Second signal array.
        return_tau (bool): If True, returns the lag at which the maximum correlation occurs.
        normalize (bool): If True, normalizes the cross-correlation to lie between -1 and 1.

    Returns:
        tuple or np.ndarray: If return_tau is True, returns a tuple (cross-correlation array, lag at max correlation).
                             If return_tau is False, returns just the cross-correlation array.

    Raises:
        ValueError: If the time stamps are not equally spaced and normalization is required.
    """
    # Calculate linear cross-correlation
    linearcorrelation = correlate(x, y, mode='full')

    # Calculate lags assuming equidistant time stamps
    if np.any(np.diff(np.diff(ts)) != 0):  # Check if spacing is not fixed
        raise ValueError("Time stamps must be equally spaced for this calculation.")
    
    lags = np.arange(-len(x) + 1, len(x)) * np.mean(np.diff(ts))
    best_lag = lags[np.argmax(linearcorrelation)]
    
    if normalize:
        autocorr_x = correlate(x, x, mode='full')
        autocorr_y = correlate(y, y, mode='full')
        normalization = np.sqrt(autocorr_x[len(x)-1] * autocorr_y[len(y)-1])
        linearcorrelation /= normalization
    
    if return_tau:
        return lags, linearcorrelation, best_lag
    else:
        return lags, linearcorrelation


def partial_covariance(x, y, z):
    """
    Computes the partial covariance between two variables x and y, controlling for the variable z.
    This is done by first regressing x and y on z independently, then calculating the residuals,
    and finally computing the covariance of these residuals.

    Parameters:
        x (array_like): The first variable array.
        y (array_like): The second variable array.
        z (array_like): The control variable array used to account for its influence on x and y.

    Returns:
        float: The partial covariance between x and y, adjusted for the influence of z.

    Examples:
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        z = np.array([1, 3, 5, 7, 9])
        print(partial_covariance(x, y, z))  # Output might vary based on the actual computations.
    """
    # Perform linear regression of z on x and y to find the best fitting line
    linreg_z2x = np.polyfit(z, x, 1)
    linreg_z2y = np.polyfit(z, y, 1)

    # Calculate residuals: differences between actual data and the predictions
    resid_x = x - np.poly1d(linreg_z2x)(z)
    resid_y = y - np.poly1d(linreg_z2y)(z)

    # Calculate and return the covariance matrix of the residuals
    pcov = np.cov(resid_x, resid_y)
    return pcov


def normalize ( y, x=None, kind='max'):
    if kind == 'max':
        return y/np.nanmax(y)

def quickfit ( predict_fn, x, y, u_y=None, u_x=None, bounds=None, fit_intrinsic_scatter=False ):
    if u_y is None:
        u_y = 0.01*y
        
    fitter = BaseInferer ()
    fitter.set_predict(predict_fn)
    fitter.set_loglikelihood(fitter.define_gaussianlikelihood(fitter.predict,fit_intrinsic_scatter))
    fitter.set_bounds(bounds)
    fitter.set_uniformprior(fitter.bounds)
    data = (x,y,u_y, u_x)
    fitter.run(data)
    return fitter, data

def closedform_leastsq (x,y):
    flat = np.ones(x.shape[0])
    w = np.hstack([flat.reshape(-1,1),x.reshape(x.shape[0],-1)])
    leastsquares_soln = np.matmul(np.matmul(np.linalg.inv(np.matmul(w.T,w)),w.T),y)
    #leastsquares_soln = np.matmul(np.matmul(np.linalg.inv(np.matmul(w.T,w)),w.T),y)
    #offset = np.mean(y - (w*leastsquares_soln).flatten(),axis=0)
    return leastsquares_soln

def logistic_fn ( var, a, mu, s, floor ):
    pq = a / (1. + np.exp(-(var-mu)/s) ) + floor
    return pq

def logistic_loglikelihood ( theta, data ):
    var,labels,weights = data
    #a, mu, s, floor = theta
    #pq = a / (1. + np.exp(-(var-mu)/s) ) + floor
    pq = logistic_fn(var, *theta)
    lnP = np.sum(weights*np.where(labels==1, np.log(pq), np.log(1.-pq)))
    return lnP   