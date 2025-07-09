import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from .functions import log_multivariate_gaussian, fast_log_multivariate_gaussian
from . import sampling

class ExtremeDeconvolution ():
    def __init__ ( self, ndim, ncomponents, mean_bounds=None, amplitude_floor=0., enforce_order=True ):
        self.ndim = ndim
        self.ncomponents = ncomponents
        self.mean_bounds = mean_bounds
        self.amplitude_floor = amplitude_floor
        self.enforce_order = enforce_order

    @property
    def nparams ( self ):
        nalpha = self.ncomponents - 1
        nmean = self.ndim*self.ncomponents
        nvar = self.ndim * (self.ndim+1) // 2 * self.ncomponents #  D * (D + 1) // 2
        #return int(self.ncomponents*(1. + self.ndim + self.ndim**2))
        return nalpha+nmean+nvar

    def convert_params ( self, theta ):
        ndim = self.ndim
        ncomponents = self.ncomponents
          
        alphas = theta[:(ncomponents-1)]
        alphas = np.concatenate([alphas, [1. - np.sum(alphas)]])

        mstart = ncomponents-1
        means = theta[mstart:mstart + ncomponents*ndim].reshape(ncomponents, ndim)        

        vstart = mstart + ncomponents*ndim
        covmatrices = sampling.vecs_to_cov_matrices ( theta[vstart:], self.ndim )  
        return alphas, means, covmatrices
        
    def lnlikelihood ( self, theta, X, u_X, use_numba=True ):
        alphas, means, covmatrices = self.convert_params(theta)
        ncomponents = len(alphas)
        nsamp = X.shape[0]
    
        lnl_ij = np.zeros([nsamp, ncomponents])
        
        for cdx in range(ncomponents):
            var = covmatrices[cdx]
            mean = means[cdx]
            amplitude = alphas[cdx]
    
            tarray = u_X + var 
            if use_numba:
                eval_fn = fast_log_multivariate_gaussian
            else:
                eval_fn = log_multivariate_gaussian        
            lpoints = eval_fn(X, mean, tarray) + np.log(amplitude) # ln(a*N)
            
            lnl_ij[:,cdx] = lpoints

        lnl_i = logsumexp(lnl_ij, axis=-1) # \\ sum over GMM components for each observation
        lnl = np.sum(lnl_i) # \\ sum over observations
    
        return lnl
    
    def lnlikelihood_vectorized(self, theta, X, u_X):
        alphas, means, covs = self.convert_params(theta)
        N, D = X.shape
        K = len(alphas)

        log_alphas = np.log(alphas)
        log_likelihoods = np.empty((N, K))

        for k in range(K):
            mu = means[k]                      # (D,)
            cov_k = covs[k]                    # (D, D)
            cov_eff = u_X + cov_k              # (N, D, D)

            # Efficient vectorized Mahalanobis + logdet per sample
            diff = X - mu                      # (N, D)

            # Compute inverse and logdet per sample
            inv_cov = np.linalg.inv(cov_eff)   # (N, D, D)
            logdet = np.linalg.slogdet(cov_eff)[1]  # (N,)

            # Mahalanobis term: xᵀ Σ⁻¹ x
            tmp = np.einsum('ni,nij->nj', diff, inv_cov)  # (N, D)
            mahal = np.einsum('nj,nj->n', tmp, diff)      # (N,)

            log_prob = -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)
            log_likelihoods[:, k] = log_prob + log_alphas[k]

        return np.sum(logsumexp(log_likelihoods, axis=1))
        
            
    def lnprior ( self, theta ):
        ndim = self.ndim
        ncomponents = self.ncomponents

        alphas, means, covmatrices = self.convert_params(theta)
        if (alphas<self.amplitude_floor).any():
            self.exitcode = 400
            return -np.inf
        amp_sum = np.sum(alphas)
        if not abs(amp_sum - 1.) < 0.05:
            self.exitcode = 100
            return -np.inf
        if self.enforce_order and (np.diff(alphas) > 0.).any():
            self.exitcode = 101
            return -np.inf
        
        if self.mean_bounds is not None:
            ll = self.mean_bounds[0] #[8., 12.5]
            ul = self.mean_bounds[1] #[10., 13.5]
            bound_array = np.repeat(ll, ncomponents).reshape(ncomponents,ndim, order='F')
            if ((means - bound_array) < 0).any(): 
                self.exitcode = 200               
                return -np.inf
            
            bound_array = np.repeat(ul, ncomponents).reshape(ncomponents,ndim, order='F')
            if ((means - bound_array) > 0).any():   
                self.exitcode =201             
                return -np.inf
        
        for var in covmatrices:            
            if (np.diag(var) < 0).any():
                self.exitcode = 300
                return -np.inf
            elif (np.linalg.eigvals(var) < 0.).any():
                self.exitcode = 301
                return -np.inf
            elif (abs(var) > 5.).any():
                self.exitcode = 302
                return -np.inf
        
        return 0.

    def lnprob ( self, theta, X, u_X ):
        lnpr = self.lnprior(theta)
        if lnpr < 0.:
            return lnpr

        lnl = self.lnlikelihood(theta, X, u_X)
        return lnpr + lnl

    def sample ( self, theta, nsample ):
        ncomponents = self.ncomponents
        ndim = self.ndim
        amplitudes, means, covs = self.convert_params(theta)
        mvn_j = [ multivariate_normal(mean=means[j], cov=covs[j]) for j in range(ncomponents) ]
        
        V = []
        labels = []
        for jdx in range(ncomponents):
            draws = int(np.ceil(nsample * amplitudes[jdx]))
            src = mvn_j[jdx]
            V.extend(src.rvs(size=draws))
            labels.extend(np.full(draws, jdx))
        V = np.array(V)   
        V = V[:nsample]
        labels = np.array(labels)[:nsample]
        return V, labels
    
    def observe ( self, theta, X, u_X ):
        nsample = len(u_X)
        V,labels = self.sample ( theta, nsample )        
        
        # matching off of first dim  
        matched_V = np.zeros_like(X)
        matched_u_X = np.zeros_like(u_X)
        for idx in range(len(X)):
            matched_V[idx,0] = X[idx,0]
            matched_V[idx,1:] = V[np.argmin(abs(X[idx,0] - V[:,0])),1:]
            matched_u_X[idx] = u_X[idx]
         
        pred_X = np.zeros_like(X)
        for idx in range(nsample):        
            pred_X[idx] = multivariate_normal(mean=matched_V[idx], cov=matched_u_X[idx] ).rvs()

        return pred_X, labels


def plot_chains (chain, truth=None, labels=None, estimate=None):
    import matplotlib.pyplot as plt

    nparams = chain.shape[-1]
    nwalkers = chain.shape[1]
    
    nrows = int(np.ceil(np.sqrt(nparams)))
    ncols = int(np.ceil(nparams/nrows))
    fsize = 2.5
    fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols*fsize*1.5, nrows*fsize) )
    faxarr = axarr.flatten()

    for pdx in range(nparams):
        for ndx in range(nwalkers):
            faxarr[pdx].plot(chain[:,ndx,pdx], color='k', alpha=0.1)
        if truth is not None:
            faxarr[pdx].axhline(truth[pdx], color='r', lw=3, ls='--' )
        if estimate is not None:
            faxarr[pdx].axhline(estimate[pdx], color='lightgrey', lw=3, ls=':' )
        if labels is not None:
            faxarr[pdx].set_ylabel(labels[pdx])
        

    for ddx in range(pdx+1, len(faxarr)):
        fig.delaxes(faxarr[ddx])    
        

def bic ( maxlnp, nobs, nparam ):
    return nparam * np.log(nobs) - 2.*maxlnp