import numpy as np

def meanerr2(x,sigma):
    
    n=len(x)  
    weight=1./sigma**2
    
    if n == 1:
        xmean  = x[0]
        xerr = (1.0/weight)**0.5
        xsigma = 0.
        xsigma2 = 0.
        xmedian = 0.
        xsigma_from_median = 0.
        xsigma_from_median2 = 0.
        xerr_median = 0.
    else:
        summ = sum(weight)
        if summ == 0.0:
            print('MEANERR2: summ is zero.')
        summx   = sum(weight*x)
        xmean  = summx/summ
        xerr = (1./summ)**0.5
        
        #Sigma weigthed Bevington 4.23: WEIGHTED SAMPLE VARIANCE
        xsigma = ((n/float(n-1)) * sum(weight*(x-xmean)**2)/summ)**0.5
        #simple sigma
        xsigma2 = ((1./float(n-1)) * sum((x-xmean)**2))**0.5
        xmedian=np.median(x)
        xsigma_from_median = ((n/float(n-1)) * sum(weight*(x-xmedian)**2)/summ)**0.5
        xsigma_from_median2 = ((1./float(n-1)) * sum((x-xmedian)**2))**0.5
        xerr_median=xsigma_from_median/(n)**0.5
        
    return (xmean,xerr,xsigma,xsigma2,xmedian,xsigma_from_median,xsigma_from_median2,xerr_median)
