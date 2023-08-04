
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import extinction


class BoxImage ( dict ):
    @property    
    def array ( self ):
        return np.array( [self[key] for key in self.keys() ] )
    
    def __setitem__(self, __key, __value):
        setattr(self, __key, __value)
        return super().__setitem__(__key, __value)
    
    
        
class FilterCurve ( object ):
    def __init__ ( self, name, transmission, filter_type ):
        self.name = name
        self.transmission = transmission
        self.filter_type = filter_type

    @property
    def lambda_mean ( self ):
        filt = self.transmission
        top = np.trapz(filt[:,1]*filt[:,0], filt[:,0])
        bot = np.trapz(filt[:,1], filt[:,0])
        return top/bot

    def interpolate_to_wv ( self, wv ):
        filt = self.transmission
        fintp = np.interp(wv, filt[:,0], filt[:,1])
        fintp[(wv<filt[0,0])|(wv>filt[-1,0])] = 0.
        return fintp
    
class FilterBase ( object ):
    def load_filtercurves ( self,
                            fnames=['FUV','NUV'],#,'g','r','i','z','y'], #'FUV','NUV',
                            fpaths=['GALEX_GALEX.FUV.dat','GALEX_GALEX.NUV.dat'],
                            #fpaths=['GALEX.FUV','GALEX.NUV','HSC-g.txt','HSC-r2.txt',
                            #        'HSC-i2.txt','HSC-z.txt','HSC-Y.txt'], # 'GALEX.FUV','GALEX.NUV',
                            ftype=None, # 'photon','photon',
                            #is_global=[True,True,False,False,False,False],
                            prefix_path='../data/filter_curves/'):
        filter_d = {}
        for ix,(name,path) in enumerate(zip(fnames,fpaths)):
            trans = np.genfromtxt(f'{prefix_path}/{path}')
            if ftype is None:
                qf = 'photon'
            else:
                qf = ftype[ix]
            filter_d[name] = FilterCurve ( name, trans, qf )
        self.filter_curves = filter_d        
        self.bands = fnames


class MultiBandImage (object):
    def __init__ ( self, bands, zpt, pixscale ):
        self.bands = bands
        if isinstance(zpt, float):
            self.zpt = np.ones(len(bands))*zpt
        else:
            self.zpt = np.asarray(zpt)
        self.pixscale = pixscale
        self.image = BoxImage ()
        self.variance = BoxImage ()
        self.mask = BoxImage ()
        self.wcs = {}
        
    def sky2pix ( self, ra, dec, bandpass):
        coord = np.array([ra,dec]).reshape(1,2)
        cwcs = self.wcs[bandpass]
        pixcoord = cwcs.all_world2pix ( coord, 0 ).flatten()
        return pixcoord

    def make_cutout ( self, ra, dec, deltaRA, deltaDEC ):  
        '''
        Produce a cutout of size deltaRA (in deg) and deltaDEC (in deg)
        '''      
        bands = self.bands
        
        width = deltaRA * (self.pixscale/3600.)**-1 # deg * (arcsec/pix * deg/arcsec)**-1
        height = deltaDEC * (self.pixscale/3600.)**-1 
        
        cutout = MultiBandImage (self.bands, self.zpt, pixscale=self.pixscale)
        for ix, key in enumerate(bands):
            centralcoord = self.sky2pix ( ra, dec, key )

            ymin = int(centralcoord[1] - height/2.)
            ymax = int(centralcoord[1] + height/2.)
            xmin = int(centralcoord[0] - width/2.)
            xmax = int(centralcoord[0] + width/2.)
            
            cutout.image[key] = self.image[key][ymin:ymax,xmin:xmax]
            cutout.variance[key] = self.variance[key][ymin:ymax,xmin:xmax]
            cutout.mask[key] = self.mask[key][ymin:ymax,xmin:xmax]
        return cutout        


def cutout_from_fits ( name, source_name, cutout_dir):    
    #from musulsb.image import MultiBandImage   
    cutout = MultiBandImage ( bands='grizy', zpt=27., pixscale=0.168 )
    for band in cutout.bands:
        fname = f'{cutout_dir}/coadds_{source_name}/fmt_{name}_{band}.fits'
        if not os.path.exists(fname):
            print(f'[OSError] {name} does not have imaging in HSC-{band}!')
            continue
        fim = fits.open(fname)
        cutout.image[band] = fim[1].data.byteswap().newbyteorder()
        cutout.mask[band]  = fim[2].data.byteswap().newbyteorder()
        cutout.variance[band] = fim[3].data.byteswap().newbyteorder()
        
        pname = f'{cutout_dir}/psfs_{source_name}/psf_{name}_{band}.fits'
        if os.path.exists(pname):
            psf = fits.open(pname)
            cutout.psf[band] = psf[0].data
        else:
            print(f'[OSError] {name} does not have a PSF model for HSC-{band}!')
    cutout.name = name
    cutout.source_name = source_name
    return cutout

class GalacticExtinction ( FilterBase ):
    def __init__(self, av=None, extinction_path=None, filter_directory='./' ):
        if (av is not None) and (extinction_path is not None):
            raise ValueError ("Cannot specify both AV and an extinction table!")
        elif av is not None:
            self.av = av
        else:
            dust_raw = open(extinction_path,'r').readlines()
            dust = pd.read_csv(extinction_path, skiprows=16, 
                            delim_whitespace=True, names=[ dr[1:] for dr in dust_raw[13].split() ][1:-1])
            dust_indices = pd.read_csv(extinction_path.replace('extinction.tbl','dustmap_indexkey.csv'), index_col=0)
            dust.index = dust_indices.index
            self.galext = dust 
            av = self.galext.loc[str(name), 'AV_SandF'] #XXX
            self. av = av
        self.load_filtercurves(prefix_path=filter_directory)
    
    def get_Alambda ( self, Av, filter_name, Rv=3.1 ):
        '''
        Using the package extinction's implementation of the Fitzpatrick+1999 Galactic
        extinction curve, compute A_lambda over a broadband filter.
        '''
        if filter_name not in self.filter_curves.keys():
            raise NameError (f"{filter_name} is not a filter that has been loaded!")
        cfilt = self.filter_curves[filter_name]

        ftype = cfilt.filter_type
        if ftype == 'photon':
            fn = lambda w,f,t: np.trapz(t*f*w,w)
        elif ftype == 'energy':
            fn = lambda w,f,t: np.trapz(t*f,w)

        wv = cfilt.transmission[:,0]
        Alambda_inst = extinction.fitzpatrick99 ( wv, Av, Rv )    
        Alambda_eff = fn (wv, Alambda_inst, cfilt.transmission[:,1] )
        Alambda_eff /= fn(wv, np.ones_like(wv), cfilt.transmission[:,1] )
        #keff = fn(wv, kc, cfilt.transmission[:,1])/fn(wv,np.ones_like(kc),cfilt.transmission[:,1])
        return Alambda_eff #keff * Av / Rv
    
    def deredden ( self, name, filter_name, Rv=3.1, verbose=False ):
        av = self.av 
        
        if verbose:
            print(f'[GalExt] extinction over is A_V={av}')
            
        alambda = self.get_Alambda ( av, filter_name, Rv )
        factor = 10**(0.4*alambda)
        return factor
    
    