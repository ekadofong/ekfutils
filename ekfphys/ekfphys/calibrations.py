import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as co


def vac2air ( vacwl ):
    '''
    IAU standard Morton 1991
    AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
    '''
    return vacwl/(1.0 + 2.735182e-4 + 131.4182 / vacwl**2 + 2.76249E8 / vacwl**4)

def convert_imf ( value, imf_in, imf_out='kroupa', imf_power=1, scale='log' ):
    """
    Convert a value between different Initial Mass Function (IMF) normalization schemes.

    This function allows conversion of a value (such as stellar mass) from one IMF normalization scheme to another.
    The conversion is based on the power-law relationship between different IMF schemes.

    Parameters:
        value (float or array-like): The value to be converted.
        imf_in (str): The IMF normalization scheme of the input value. Options are 'salpeter', 'kroupa', or 'chabrier'.
        imf_out (str, optional): The IMF normalization scheme to convert to. Options are 'salpeter', 'kroupa', or 'chabrier'.
            Default is 'kroupa'.
        imf_power (float, optional): The power of the power-law relationship between IMF normalization schemes.
            Default is 1.

    Returns:
        float or array-like: The converted value in the new IMF normalization scheme.
    """    
    logfactor_to_salpeter = {'salpeter':0., 
                             'kroupa':-np.log10(0.61), # gamma
                             'chabrier':-np.log10(0.66) # alpha
                             }
    
    if scale == 'log':
        conversion = logfactor_to_salpeter[imf_in.lower()] - logfactor_to_salpeter[imf_out.lower()]
        return value + imf_power * conversion
    elif scale == 'linear':
        conversion = 10.**(logfactor_to_salpeter[imf_in.lower()] - logfactor_to_salpeter[imf_out.lower()])
        return value * conversion**imf_power
    
def LHa_from_EW ( EW, Mr, z=0., balmer_dec=2.86, ew_c=2.5, ):
    wv_ha = 6565.*(1.+z)
    prefactor = 3e25 * u.erg/u.s
    ha_lum = prefactor * (EW+ew_c) / wv_ha**2 * 10.**( (Mr - 34.1)/-2.5 ) * ( balmer_dec / 2.86 ) ** 2.36
    return ha_lum

def EWfromLHa(LHa, Mr, z=0., balmer_dec=2.86, ew_c=2.5, ):
    wv_ha = 6565.*(1.+z)
    prefactor = 3e25 * u.erg/u.s
    phot_factor = 10.**( (Mr - 34.1)/-2.5 ) * ( balmer_dec / 2.86 ) ** 2.36
    combined_ew = LHa / prefactor  * wv_ha**2  / phot_factor   
    haew = combined_ew - ew_c
    return haew

def LNUV2SFR_IP ( lnuv ):
    '''
    Following Eq. 3 of Iglesias-Paramo 2006, corrected for Kroupa IMF
    '''
    return lnuv/(10.**9.33) * 0.66

def SFR_UV2Ha ( logsfr_uv, conversion='lee09'):
    '''
    Following Lee+09, but use with caution
    '''
    if conversion == 'lee09':
        #logsfr_ha = (logsfr_uv + 0.2)/0.79
        m = 0.32
        b = 0.37
        
        tp = -1.5
        tipping_point = tp*(1.-m) - b
        print(tipping_point)
        logsfr_ha = np.where(
            logsfr_uv < tipping_point,
            (logsfr_uv+b)/(1.-m),
            -0.13 + logsfr_uv
        )
    elif conversion == 'saga':
        logsfr_ha = (logsfr_uv + 0.631)/0.733
    return logsfr_ha

def LUV2SFR_K98 ( uv_luminosity ):
    '''
    THIS IS WITH SALPETER IMF
    
    SFR(Modot yr-1) = 1.4 x 10-28 Lnu(UV) (erg s-1 Hz-1). (3)
    '''
    prefactor = 1.4e-28
    if hasattr ( uv_luminosity, 'unit' ):
        uv_luminosity = uv_luminosity.to(u.erg/u.s/u.Hz).value
    return prefactor * uv_luminosity

def LHa2SFR_K98 ( ha_luminosity ):
    '''
    THIS IS WITH SALPETER IMF
    
    Kennicut + 1998
    '''
    if hasattr ( ha_luminosity, 'unit'):
        ha_luminosity = ha_luminosity.to(u.erg/u.s).value
    return ha_luminosity / 1.26e41

def SFR2LHa_K98 ( sfr ):
    '''
    See above
    '''
    return 1.26e41 * sfr

def SFR2LHa_dlR ( sfr ):
    beta_dlr = (1.26e41*1.8)**-1 # see section 5.2 of de los Reyes+2015
    return sfr / beta_dlr

def LHa2SFR_calzetti ( ha_luminosity ):
    '''
    From https://ned.ipac.caltech.edu/level5/Sept12/Calzetti/Calzetti1_2.html, eq. 1.10
    Assumes a Kroupa IMF
    '''
    if hasattr ( ha_luminosity, 'unit' ):
        ha_luminosity = ha_luminosity.to(u.erg/u.s).value
        return_unit = True
    else:
        return_unit = False
    if return_unit:
        return ha_luminosity * 5.5e-42 * u.Msun/u.yr
    else:
        return ha_luminosity * 5.5e-42

def SFR2LHa_calzetti ( ha_sfr ):
    '''
    See above
    '''
    return ha_sfr / 5.5e-42

def SFR2LHa_brinchmann ( ha_sfr ):
    '''
    Brinchmann+04 average adopted value for a Kroupa IMF,
    SFR = LHa/10^41.28
    '''
    return ha_sfr * 10**41.28

def LHa2SFR_lee ( ha_lum ):
    '''
    From Lee+09, low LHa luminosity correction to SFRs
    '''
    pivot = 2.5e39
    if isinstance(ha_lum, float):
        ha_lum = np.array(ha_lum)
    
    lowluminosity = ha_lum <= pivot
    sfr = ha_lum.copy()
    sfr[~lowluminosity] = LHa2SFR_K98(ha_lum[~lowluminosity])
    sfr[lowluminosity] = 0.62 * np.log10(7.9e-42 * ha_lum[lowluminosity]) - 0.47
    return sfr

def LHa2SFR_recompute ( ha_lum ):
    '''
    I'm recomputing everything using Jeong-Gyu's SB99 calculations of
    specific ionizing photon rate from my 2020b paper
    
    LHa = alpha_eff/alpha_B Q_H hc/lambda_Ha
    Q_H = q_H SFR dt 
    SFR = alpha_B/alpha_eff LHa/(q_H dt hc/lambda_Ha )
    '''
    beta = 4.87e-42
    return ha_lum * beta

def SFR2LHa_recompute ( ha_sfr ):
    beta = 4.87e-42
    return ha_sfr/beta

def LHa2SFR ( ha_lum ):
    return LHa2SFR_calzetti(ha_lum)

def SFR2LHa ( ha_sfr ):
    return SFR2LHa_calzetti ( ha_sfr )

def SFRK2C ( SFR_kennicutt ):
    '''
    Convert form a K98 SFR(LHa) prescription to a NED Calzetti SFR(LHa)
    '''
    alpha_c = 5.5e-42
    alpha_k = (1.26e41)**-1
    return alpha_c/alpha_k * SFR_kennicutt

def log_uncertainty ( n, u_n ):
    # X = log10(n)
    # v_X = (dX/dn)^2 v_n
    #     =  (ln10 n)^-2 v_n
    return u_n/(np.log(10.)*n)

def logratio_uncertainty ( n, u_n, d, u_d ):
    # X = log10 ( n / d)
    # v_X = (1/(n/d ln10) 1/d)^2 v_n + ( 1/(n/d ln10) n/d^2 )^2 v_d
    # v_X = (d/[n ln10])^2 * ( v_n / d^2 + n^2/d^4 v_d )
    #     = (n ln10)^-2 * ( v_n + n_2/d^2 v_d )
    return np.sqrt ( (n * np.log(10.))**-2. * ( u_n**2  + n**2/d**2 * u_d**2 ) )

def R23 (f_OII3727, f_OIII4959, f_OIII5007, f_Hbeta, uncertainties=None ):
    r23 =  np.log10((f_OII3727+f_OIII4959+f_OIII5007)/f_Hbeta)
    if uncertainties is not None:
        numerator_var = uncertainties[0]**2 + uncertainties[1]**2 + uncertainties[2]**2
        osum = f_OII3727+f_OIII4959+f_OIII5007
        u_r23 = logratio_uncertainty ( osum, np.sqrt(numerator_var), f_Hbeta, uncertainties[3] )
        return r23, u_r23
    return r23

def O32 ( f_OII3727, f_OIII4959, f_OIII5007 ):
    o32 = (f_OIII4959+f_OIII5007)/f_OII3727
    return o32

def O3N2 ( f_OIII5007, f_Hbeta, f_NII6583, f_Halpha, uncertainties=None ):
    '''
    Note that we are going to return log10(O3N2)
    '''
    o3n2 = (f_OIII5007/f_Hbeta) / (f_NII6583/f_Halpha)
    logarithmic_calibrator = np.log10(o3n2)
    
    if uncertainties is not None:
        variances = log_uncertainty(f_OIII5007, uncertainties[0])**2
        variances += log_uncertainty(f_Hbeta, uncertainties[1])**2
        variances += log_uncertainty(f_NII6583, uncertainties[2])**2
        variances += log_uncertainty(f_Halpha, uncertainties[3])**2
        u_calib = np.sqrt(variances)
    else:
        u_calib = None
    return logarithmic_calibrator, u_calib

def N2 ( f_NII6583, f_Halpha, u_f_NII6583=None, u_f_Halpha=None ):
    ratio = np.log10(f_NII6583/f_Halpha)
    if u_f_NII6583 is not None:
        u_ratio = logratio_uncertainty ( f_NII6583, u_f_NII6583, f_Halpha, u_f_Halpha )
        return ratio, u_ratio
    else:
        return ratio
    
def Pionization ( f_OII3727, f_OIII4959, f_OIII5007 ):
    # https://iopscience.iop.org/article/10.1086/432408/pdf
    R3 = f_OIII4959 + f_OIII4959
    R2 = f_OII3727
    return R3/(R2+R3)

def PT05 ( R23, P ):
    upper = (R23 + 726.1 + 842.2*P + 337.5*P**2)/(85.96 + 82.76*P + 43.98*P**2 + 1.793*R23)
    lower = (R23 + 106.4 + 106.8*P - 3.4*P**2)/(17.72 + 6.6*P + 6.95*P**2 - 0.302*R23 )
    return lower

def ionization_parameter ( logOHp12, f_OII3727, f_OIII4959, f_OIII5007):
    y = np.log10(O32(f_OII3727, f_OIII4959, f_OIII5007))
    t0 = 32.81 - 1.153*y**2 + logOHp12*(-3.396-0.025*y+0.1444*y**2)
    t1 = 4.603 - 0.3119*y -0.163*y**2 + logOHp12*(-0.48+0.0271*y+0.02037*y**2)
    logq = t0/t1
    return logq

def M13_N2 (N2, u_N2=None):
    metallicity =  8.743 + 0.462 * N2   
    if u_N2 is not None:
        u_met = 0.462 * u_N2
        return metallicity, u_met
    else: 
        return metallicity

    
    
def PMC09 (o3n2):
    return 8.74 - 0.31 * np.log10(o3n2)

def brownN2 ( N2, logsfr, logmstar ):
    avg_logsSFR = 283.728 - 116.265*logmstar + 17.4403*logmstar**2 - 1.17146*logmstar**3 + 0.0296528*logmstar**4
    logsSFR = logsfr - logmstar
    
    delta_logsSFR = logsSFR - avg_logsSFR
    oh_brown = 9.12 + 0.58 * N2 - 0.19 * delta_logsSFR        
    return oh_brown

def T04 ( f_OII3727, f_Hbeta, f_OIII ):
    pass

def Z94 ( logr23, u_logr23=None ):
    carray = np.array([-0.333, -0.207, -0.202, -0.33, 9.265])
    logOHp12 = np.dot(np.vander ( logr23, 5 ), carray)
    if u_logr23 is not None:
        u_met = (carray[3] + 2.*carray[2]*logr23 * 3.*carray[1]*logr23**2 + 4.*carray[0]*logr23**3)*u_logr23
    else:
        u_met = None
    if hasattr ( logr23, 'index'):
        logOHp12 = pd.Series(logOHp12, index=logr23.index)
    return logOHp12, u_met

def M13 ( log_o3n2, u_log_o3n2=None ):
    met = 8.533 - 0.214 * log_o3n2
    if u_log_o3n2 is not None:
        u_met = 0.214 * u_log_o3n2
    else:
        u_met = None
    return met, u_met

def P03 (r23, pion):
    '''
    Pilyugin+2003 Equation 9
    '''
    top = r23 + 54.2 + 59.45*pion + 7.31*pion**2
    bottom = 6.07 + 6.71*pion + 0.37*pion**2 + 0.243*r23
    return top/bottom

def KD03 (f_OII3727, f_OIII4959, f_OIII5007, f_Hbeta, learning_rate=0.03,  itermax=100):
    '''
    Kobulnicky & Kewley + 04
    '''
    r_R23 = R23(f_OII3727, f_OIII4959, f_OIII5007, f_Hbeta,)
    x = np.log10(r_R23)
    fn = lambda z: abs(9.4 + 4.65*x -3.17*x**2 - ionization_parameter(z, f_OII3727, f_OIII4959, f_OIII5007)*(0.272+0.547*x-0.513*x**2) - z)

    z_old = 8.*np.ones_like(x)
    A_old = fn(z_old)
    grad0 = -0.5
    learning_rate = 0.01

    z_new = z_old - learning_rate*grad0
    A_new = fn(z_new)
    #path = np.zeros(100)*np.NaN
    #grad = path.copy()
    #zx = path.copy()
    niter = 0
    while True:
        #print(f'{(A_new-A_old).mean()} {(z_new-z_old).mean()}')
        roughgrad = (A_new-A_old)/(z_new-z_old)
        if abs(A_new.mean()) <= 0.01:
            break
        elif niter >= itermax:
            break
            
        z_old = z_new
        A_old = fn(z_old)
        z_new = z_old - roughgrad * learning_rate
        A_new = fn(z_new)
        #path[niter] = A_old.mean()
        #grad[niter] = roughgrad.mean()
        #zx[niter] = z_new.mean()
        niter += 1
    return z_new
    
def PP04 (n2):
    '''
    Pettini & Pagel 2004
    '''
    cc = np.array([0.32, 1.26, 2.03, 9.37])

    logOHp12 = np.dot(np.vander ( n2, 4 ), cc)
    return logOHp12

