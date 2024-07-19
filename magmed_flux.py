import numpy as np
import sys
from meanerr2 import meanerr2

def magmed(mag2, err2=None, median_or_mean=False):
     
    # do_biweight
    # 1: YES, WITH MEDIA AND SIGMA BEVINGTON
    # 2: YES, WITH MEDIAN AND SIGMA MEDIAN WEIGHTED
    
    if err2 is None:
        flux=10.**(-mag2/2.5)
        magmeanflux = -2.5 *np.log10(sum(flux)/len(flux))

        return magmeanflux
    
    else:
  
        flux = 10**(-mag2/2.5)
        errflux = flux*np.log(10)*.4*err2
        
        sss = meanerr2(flux,errflux)
        
        if ~median_or_mean:
            meanflux = sss[4]
            errmeanflux = sss[5]
        else:
            meanflux = sss[0]
            errmeanflux = sss[1]

        magmeanflux = -2.5 * np.log10(meanflux)
        error_on_flux = errmeanflux*2.5/(np.log(10)*meanflux)

        return [magmeanflux,error_on_flux]
