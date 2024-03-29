from typing import ItemsView
import numpy as np
from astropy import wcs
from astropy import units as u
from ekfparse import query
from . import image

_GALEX_FWHM = {'FUV':4.5*u.arcsec, 'NUV':5.6*u.arcsec} # \\ https://asd.gsfc.nasa.gov/archive/galex/Documents/instrument_summary.html

def galex_cts2flux ( cts, band ):
    if band == 'FUV':
        conversion = 1.4e-15
    elif band == 'NUV':
        conversion = 2.06e-16
    else:
        raise ValueError (f"{band} not recognized as a Galex band!")    
    return cts * conversion

def galex_cts2mag ( cts, band ):
    if band == 'FUV':
        zeropoint = 18.82    
    elif band == 'NUV':
        zeropoint = 20.08
    return -2.5*np.log10(cts) + zeropoint

def convert_zeropoint ( flux, zp_orig, zp_new ):
    expterm = (zp_orig-zp_new)/-2.5
    return flux * 10.**expterm

class MultiBandWCS (dict):
    def __init__ ( self ):
        self.bands = []
        
    def __setitem__(self, key, item):
        setattr(self, key, item)
        self.bands.append(key)
        
    def __getitem__ ( self, key):           
        return getattr(self,key)
    
    def get_pixscale ( self, band ):
        cwcs = self[band]
        pixscale = abs(cwcs.pixel_scale_matrix[0,0]) *  3600.
        return pixscale
    
    #def keys ( self )

class Imaging ( object ):
    def sky2pix ( self, ra, dec, bandpass):
        coord = np.array([ra,dec]).reshape(1,2)
        cwcs = self.wcs[bandpass]
        galex_pixcoord = cwcs.all_world2pix ( coord, 0 ).flatten()
        return galex_pixcoord

    def make_cutout ( self, ra, dec, deltaRA, deltaDEC ):        
        bands = self.bands
        
        width = deltaRA * (self.pixscale/3600.)**-1 # deg * (arcsec/pix * deg/arcsec)**-1
        height = deltaDEC * (self.pixscale/3600.)**-1 
        
        cutout = image.MultiBandImage (bands, [self._galex_zpt(x) for x in bands ], pixscale=self.pixscale)
        for ix,im in enumerate(bands):
            key = bands[ix]
            galex_centralcoord = self.sky2pix ( ra, dec, key )

            ymin = int(galex_centralcoord[1] - height/2.)
            ymax = int(galex_centralcoord[1] + height/2.)
            xmin = int(galex_centralcoord[0] - width/2.)
            xmax = int(galex_centralcoord[0] + width/2.)
            cutout.image[key] = im[0].data[ymin:ymax,xmin:xmax]
        return cutout
    
class GalexImaging ( Imaging ):
    def __init__ ( self, #cutout_id,
                  galex_bundle=None, fuv_im=None, nuv_im=None, pixscale=1.5,
                  correct_galacticextinction=True,          
                  av=None,        
                  filter_directory='./',
                  verbose=True ):
        # \\ load images
        if (galex_bundle is None) and (fuv_im is None) and (nuv_im is None):
            raise ValueError ( "Must either supply a Galex bundle or FUV and/or NUV images")        
        elif galex_bundle is None:
            self.fuv_im = fuv_im
            self.nuv_im = nuv_im            
        else:
            self.fuv_im = galex_bundle['fd']
            self.nuv_im = galex_bundle['nd']
        
        
        self.bands = []#'FUV','NUV']
        self.wcs = MultiBandWCS () 
        for name, key in zip(['FUV','NUV'],['fd','nd']):
            if galex_bundle[key] is not None:
                self.bands.append(name)
                self.wcs[name]= wcs.WCS ( galex_bundle[key][0].header )

        # \\ image properties
        self.pixscale = pixscale # arcsec / pixel     
        if self.fuv_im is None:
            self._imshape = self.nuv_im[0].data.shape
            #self._wcs = wcs.WCS ( self.nuv_im[0].header )
        else:
            self._imshape = self.fuv_im[0].data.shape
            #self._wcs = wcs.WCS ( self.fuv_im[0].header )
            
        # \\ set up galactic extinction correction
        if correct_galacticextinction:
            if av is None:
                cutout_size = self.nuv_im[0].shape
                center = self.wcs['NUV'].all_pix2world(cutout_size[0]//2, cutout_size[1]//2,1)
                av = query.get_SandFAV ( center[0], center[1] )
            self.ge = image.GalacticExtinction ( av=av, filter_directory=filter_directory )
            #self.cutout_id = cutout_id

        self.verbose = verbose 
        
    def _galex_zpt ( self, key ):
        if key=='FUV':
            return 18.82
        elif key=='NUV':
            return 20.08
        else:
            raise KeyError (f"galex band {key} not recognized!")

    def make_cutout ( self, ra, dec, deltaRA, deltaDEC ):
        '''
        Keeping GALEX cutout maker separate from base class since the
        code is already written
        '''
        #Y,X = np.mgrid[:self._imshape[0],:self._imshape[1]]
        bands = self.bands
        
        width = deltaRA * (self.pixscale/3600.)**-1 # deg * (arcsec/pix * deg/arcsec)**-1
        height = deltaDEC * (self.pixscale/3600.)**-1 
        
        cutout = image.MultiBandImage (bands, [self._galex_zpt(x) for x in bands ], pixscale=self.pixscale)
        for ix,im in enumerate([self.fuv_im, self.nuv_im]):
            key = bands[ix]
            galex_centralcoord = self.sky2pix ( ra, dec, key )

            ymin = int(galex_centralcoord[1] - height/2.)
            ymax = int(galex_centralcoord[1] + height/2.)
            xmin = int(galex_centralcoord[0] - width/2.)
            xmax = int(galex_centralcoord[0] + width/2.)
            cutout.image[key] = im[0].data[ymin:ymax,xmin:xmax]
        return cutout
    
    def do_upperlimitphotometry ( self,
                                  central_coordinates,
                                  apersize='psf',
                                  output_unit='Jy',
                                ):
        Y,X = np.mgrid[:self._imshape[0],:self._imshape[1]]

        flux_arr = np.zeros([2,2], dtype=float)
        bands = ['FUV','NUV']
        ims = [self.fuv_im, self.nuv_im]
        
        for ix,im in enumerate(ims):                    
            key = bands[ix]
            #im = ims[ix]
            
            if im is None:
                flux_arr[:,ix] = np.NaN
                continue
                        
            # \\ map sky coordinate to GALEX pixels
            coord = np.asarray(central_coordinates).reshape(1,2)
            
            cwcs = self.wcs[key]
            galex_pixcoord = cwcs.all_world2pix ( coord, 0 ).flatten()
            yoff = Y - galex_pixcoord[1]
            xoff = X - galex_pixcoord[0]
            R = np.sqrt(xoff**2 + yoff**2)
            
            # \\ define aperture
            if apersize == 'psf':
                #apersize = _GALEX_FWHM[key]
                apersize_pixel = _GALEX_FWHM[key].to(u.arcsec).value / self.pixscale 
            elif hasattr ( apersize, 'unit'):                
                apersize_pixel = apersize.to(u.arcsec).value * u.arcsec                
            else:
                apersize_pixel = apersize
            
            self.emask = R < apersize_pixel
            
            # \\ background-subtracted intensity map 
            flux_native = np.sum(im[0].data[self.emask] - im[2].data[self.emask])
            e_flux_native = np.sqrt(np.sum(im[1].data[self.emask]))
            
            if output_unit == 'native':
                flux = flux_native
                e_flux = e_flux_native  
            else:              
                if output_unit=='Jy':
                    zp_out = 8.9
                elif output_unit == 'hsc':
                    zp_out = 27.
                elif output_unit=='nanomaggy':
                    zp_out = 22.5
                flux = convert_zeropoint(flux_native, self._galex_zpt(key), zp_out)
                e_flux = convert_zeropoint(e_flux_native, self._galex_zpt(key), zp_out)

                
            # \\ Add in galactic extinction
            if hasattr ( self, 'ge'):
                factor = self.ge.deredden ( key )
            else:
                factor = 1.
            if self.verbose:
                print(f'[GalexImaging] 10^(0.4*A_{key}) = {factor:.4f}')
            flux_arr[:,ix] = [factor*flux,factor*e_flux]
        
        return flux_arr        
        
        
    def do_ephotometry ( self, 
                         central_coordinates, 
                         catparams, 
                         cat_pixscale=0.168, 
                         output_unit='Jy',
                         ellipse_size=9.,
                         geom_type='sep',
                         ):
        '''
        Perform elliptical aperture photometry on GALEX imaging data.

        Parameters:
            central_coordinates (tuple or list): Central coordinates (RA, Dec) in degrees.
            catparams (dict): Dictionary containing elliptical aperture parameters such as 'cyy', 'cxx', and 'cxy'.
            cat_pixscale (float, optional): Pixel scale of the catalog in arcseconds per pixel. Default is 0.168.
            output_unit (str, optional): Desired output flux unit. Options are 'native', 'Jy', or 'hsc'. Default is 'Jy'.
            ellipse_size (float, optional): Size of the elliptical aperture in squared units. Default is 9.

        Returns:
            None

        This function performs elliptical aperture photometry on GALEX imaging data. It calculates the background-subtracted flux
        and its uncertainty within the specified elliptical aperture for FUV and NUV bands. The central coordinates of the
        aperture, along with the elliptical aperture parameters, are provided in 'central_coordinates' and 'catparams'
        respectively. The pixel scale of the catalog can be adjusted with 'cat_pixscale'. The calculated flux is then converted
        to the desired 'output_unit' flux unit, which can be 'native', 'Jy' (Jansky), or 'hsc' (HSC flux units). The parameter
        'ellipse_size' defines the size of the elliptical aperture.

        Note: This function modifies attributes of the class instance where it is called.
        '''        
        Y,X = np.mgrid[:self._imshape[0],:self._imshape[1]]
        pix_conversion = self.pixscale / cat_pixscale
        
        flux_arr = np.zeros([2,2], dtype=float)
        bands = ['FUV','NUV']
        ims = [self.fuv_im, self.nuv_im]
        
        for ix,im in enumerate(ims):                    
            key = bands[ix]
            #im = ims[ix]
            
            if im is None:
                flux_arr[:,ix] = np.NaN
                continue
                        
            # \\ map sky coordinate to GALEX pixels
            coord = np.asarray(central_coordinates).reshape(1,2)
            
            cwcs = self.wcs[key]
            galex_pixcoord = cwcs.all_world2pix ( coord, 0 ).flatten()
            yoff = Y - galex_pixcoord[1]
            xoff = X - galex_pixcoord[0]
            
            # \\ define elliptical aperture
            if geom_type=='sep':
                cyy = catparams['cyy'] * pix_conversion**2
                cxx = catparams['cxx'] * pix_conversion**2
                cxy = catparams['cxy'] * pix_conversion**2
                ellipse = cyy*yoff**2 + cxx*xoff**2 + cxy*xoff*yoff  
                self.emask = ellipse < ellipse_size
            elif geom_type == 'desi':
                ellone = catparams['SHAPE_E1']
                elltwo = catparams['SHAPE_E2']
                raise NotImplementedError
            
            
            
            # \\ background-subtracted intensity map 
            flux_native = np.sum(im[0].data[self.emask] - im[2].data[self.emask])
            e_flux_native = np.sqrt(np.sum(im[1].data[self.emask]))
            
            if output_unit == 'native':
                flux = flux_native
                e_flux = e_flux_native  
            else:              
                if output_unit=='Jy':
                    zp_out = 8.9
                elif output_unit == 'hsc':
                    zp_out = 27.
                elif output_unit=='nanomaggy':
                    zp_out = 22.5
                flux = convert_zeropoint(flux_native, self._galex_zpt(key), zp_out)
                e_flux = convert_zeropoint(e_flux_native, self._galex_zpt(key), zp_out)

                
            # \\ Add in galactic extinction
            if hasattr ( self, 'ge'):
                factor = self.ge.deredden ( key )
            else:
                factor = 1.
            if self.verbose:
                print(f'[GalexImaging] 10^(0.4*A_{key}) = {factor:.4f}')
            flux_arr[:,ix] = [factor*flux,factor*e_flux]
        
        return flux_arr