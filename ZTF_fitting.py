"""
Performs Fourier and Spline fit of a given ZTF lightcurve
"""

import pandas as pd
from os.path import join as pathjoin
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.interpolate import interp1d
from math import isnan
from time import sleep
from itertools import repeat
from astropy.io import ascii
from astropy.table import Table

# import sklearn
# from sklearn.neural_network import MLPClassifier
# from sklearn.neural_network import MLPRegressor

# # Import necessary modules
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from sklearn.metrics import r2_score

# from astropy.modeling.spline import Spline1D
# from astropy.modeling.fitting import (SplineInterpolateFitter,
#                                       SplineSmoothingFitter,
#                                       SplineExactKnotsFitter)

class Switch(dict):
    def __getitem__(self, item):
        for key in self.keys():                 # iterate over the intervals
            if item in key:                     # if the argument is in that interval
                return super().__getitem__(key) # return its associated value
        raise KeyError(item)                    # if not in any interval, raise KeyError

def read_ZTF_Joseph(filein, preliminary_t0=False):
    df=pd.read_csv(filein, 
                    usecols=["oid","hjd","mag","magerr","catflags","filtercode",
                             "phase_qual","new_period","fap"])
    
    period_to_use = df.new_period.values[df.fap == min(df.fap)][0]
    
    t0 = 0.
    if preliminary_t0:
        df_groups=df.groupby('filtercode')
        filters=set(df.filtercode.values)
        for filter, group in df_groups:
            t0 = group.hjd.values[group.mag.values == min(group.mag.values)]
#             print(filter)
#             print(t0)
            if (filter == 'zg') | (filter == 'zr'):
                break
            
    phase=(df.hjd.values - t0) / period_to_use % 1.
    df['phase']=phase
    return df

def findblazhko_prova1(folder,subfolder_lcvs,lcv_temp,nbins,overlap=0.):
    cmap = plt.get_cmap('hsv')
    xfit=np.linspace(0.,1.,10001)
    
    df=read_ZTF_Joseph(folder+subfolder_lcvs+lcv_temp)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    df_groups=df.groupby('filtercode')
    filters=set(df.filtercode.values)
    
    color_dict={'zg':0.0,'zr':0.5,'zi':0.8}

    for filter, group in df_groups:
        if filter == 'zi':
            continue
        p1, = plt.plot(group.phase, group.mag.values,'.',label=filter,c=cmap(color_dict[filter]))
        nfas = len(group.phase)
        
    return 0

def dividibin(x,y,nbins):
    s, edges, _ = binned_statistic(x, y, statistic='mean', bins=np.linspace(0, 1, nbins))

    ys = np.repeat(s, 2)
    xs = np.repeat(edges, 2)[1:-1]

    return [[xs], [ys]]
    

def fourfit(folder,subfolder_lcvs,lcv_temp):
    cmap = plt.get_cmap('hsv')
    xfit=np.linspace(0.,1.,10001)
    
    df=read_ZTF_Joseph(folder+subfolder_lcvs+lcv_temp)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    df_groups=df.groupby('filtercode')
    filters=set(df.filtercode.values)
    
    color_dict={'zg':0.0,'zr':0.5,'zi':0.8}
    
    for filter, group in df_groups:
        p1, = plt.plot(group.phase, group.mag.values,'.',label=filter,c=cmap(color_dict[filter]))
        nfas = len(group.phase)

        switch = Switch({
            range(1, 21): 3,
            range(21, 31): 4,
            range(31, 51): 5,
            range(51, 81): 7,
            range(81, 121): 8,
            range(121, 999999): 10
            })
        
        degree=switch[nfas]

        #initialize array of coefficients (A0, phi0, A1, phi1..., offset, degree)
        a_init=[1.0,.1] * degree + [np.mean(group.mag)]
        popt, pcov = curve_fit(fourier_series, group.phase, group.mag, a_init)
            
        p2, = plt.plot(xfit, fourier_series(xfit, *popt),c='k')
        
    plt.gca().invert_yaxis()
    ax1.legend()
    plt.show()
    
    return 0

def splfit(folder,subfolder_lcvs,lcv_temp):
    cmap = plt.get_cmap('hsv')
    xfit=np.linspace(0.,1.,10001)
    
    df=read_ZTF_Joseph(folder+subfolder_lcvs+lcv_temp)
    df.sort_values('phase', inplace=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    df_groups=df.groupby('filtercode')
    filters=set(df.filtercode.values)

    color_dict={'zg':0.0,'zr':0.5,'zi':0.8}

    for filter, group in df_groups:
        p1, = plt.plot(group.phase, group.mag.values,'.',label=filter,c=cmap(color_dict[filter]))
        
        spl = Spline1D()
        fitter = SplineSmoothingFitter()
        spl2 = fitter(spl, group.phase, group.mag.values, s=2.5)

        plt.plot(xfit, spl2(xfit),c='k')
        
    plt.gca().invert_yaxis()
    ax1.legend()
    plt.show()
        
    return 0

## usare la prima per curve_fit di scipy
def fourier_series(x, *a): 
    sleep(0.003)
    degree = (len(a)-1)/2
    ret = a[-1]
    for deg in range(int(degree)):
        ret += a[deg*2] * np.cos((deg+1)*2*np.pi * x + a[1 + deg*2])
    return ret

def curve_fit_sigmaclip(fn, x, y, a, sigma, threshold=5., absolute_sigma=True, maxfev=5000):
    sleep(0.005)
    n_removed = 1
    ind_clipped = list(range(0,len(x)))
    ind_clipped_old = ind_clipped
    sss = 0
    while n_removed >= 1:
        x = x[ind_clipped]
        y = y[ind_clipped]
        sigma = sigma[ind_clipped]
#         print('n_removed:', n_removed)
        popt, pcov = curve_fit(fn, x, y, a, sigma=sigma, absolute_sigma=True, maxfev=maxfev)
#         print('popt:', popt)
        residuals = y-fn(x, *popt)
        std = np.std(residuals)
#         print('std:', std)
        
        ind_clipped = np.where(abs(residuals) < threshold * std)[0].tolist()
        n_removed = len(ind_clipped_old) - len(ind_clipped)
        
#         print(ind_clipped)
#         plt.scatter(x,y)
#         plt.plot(x,fourier_series(x,*popt))

        ind_clipped_absolute = [ind_clipped_old[i] for i in ind_clipped]
        ind_clipped_old = ind_clipped_absolute
        
        sss = sss +1
#         print(sss, n_removed)
        if sss > 200:
            assert False
        
    return popt, pcov, ind_clipped_absolute, std

def meanmag_flux(mag,err):
    sleep(0.005)
    flux = [10**(0-m/2.5) for m in mag]
    errflux = [f*np.log(10)*.4*e for f,e in zip(flux,err)]
    
    meanflux = np.mean(flux)
    errmeanflux = np.std(flux)
    
    magmeanflux = -2.5*np.log10(meanflux)
    error_on_flux = errmeanflux*2.5/(np.log(10)*meanflux)
    
    return magmeanflux, error_on_flux

# def max_gap(a):
#     """
#     Very efficient, O(n), do not need sort before
#     https://stackoverflow.com/questions/53041365/python-maximum-difference-between-elements-in-a-list
#     """
#     sleep(0.005)
#     vmin = a[0]
#     dmax = 0
#     for i in range(len(a)):
#         if (a[i] < vmin):
#             vmin = a[i]
#         elif (a[i] - vmin > dmax):
#             dmax = a[i] - vmin
#     return dmax

def max_gap(a):
    sleep(0.003)
    b=sorted(a)
    maxgap=0.
    for iii in np.arange(len(b)-1):
        if abs(b[iii+1]-b[iii]) > maxgap:
            maxgap = abs(b[iii+1]-b[iii])
            index=iii
    del b
    return maxgap

def find_epoch_tmeanrise(hjd0,x,y,xfit,yfit,meanmag,period):
    # Parto dal minimo di luce ...
    xfit = np.concatenate((xfit, xfit), axis=0)
    yfit = np.concatenate((yfit, yfit), axis=0)
    zzz = np.argmax(yfit)
    
    # ... e cerco l'intersezione con la meanmag
    while not ((yfit[zzz]>=meanmag) and (yfit[zzz+1]<=meanmag)):
        zzz = zzz + 1
    
    # ... interpolo
    fas_tmeanrising = xfit[zzz]+(xfit[zzz+1]-xfit[zzz])/(yfit[zzz+1]-yfit[zzz])*(meanmag-yfit[zzz])
        
    hjd_tmeanrising = hjd0 - (x[0]-fas_tmeanrising) * period
    
    return hjd_tmeanrising

def fourfit_simple(x, y, err, name, filter, subfolder_lcvs, findblazhko=False,
                   threshold_clipping=10., chisq_ratio_threshold=.98,
                  do_print_fourfit=False,do_plot_fourfit=True,suffix_fileout='',
                  folderout='',max_degree=10,autoselect_starting_degree=True,starting_degree=3,
                  maxgap_threshold=.2, maxfev=5000):
    sleep(0.005)
#     print(name,filter)
    chisqs=[]
    popts=[]
    stds=[]
    degrees=[]
    nfit = 1000
    xfit = np.linspace(0., 1., nfit + 1)

    nfas = len(x)

    #prima passata: tolgo i punti superoutlier
    ind_clipped = abs( y - np.median(y) < 1.)
#     print("Prima rejection: ",len(x) - len(ind_clipped[ind_clipped]))

    x = x[ind_clipped]
    y = y[ind_clipped]
    err = err[ind_clipped]

    #Per come è fatto, l'uniformity potrebbe non essere un indicatore ottimale... quindi binno le fasi
    #in bin di binsize (0.01-0.02-0.05) e calcolo l'uniformity sulle fasi binnate.
    x_binned = binned_statistic(x, x, bins=20, range=(0, 1.))[0]
    good=[not isnan(i) for i in x_binned]
    x_binned = x_binned[good]
    uniformityks_binned = uniformityKS(x_binned)
    uniformityks = uniformityKS(x)

    perc_lo = 2
    perc_hi = 98

    ampl = np.percentile(y,perc_hi) - np.percentile(y,perc_lo)
    ampl2 = .96 * (max(y) - min(y))
    preliminary_mean = (max(y) + min(y))/2.

    preliminary_ampl = ampl

#     print(len(np.where((y > perc05) & (y < perc95))[0].tolist())/len(y))
#     fraction_within_90perc = len()

    if autoselect_starting_degree == True:
        starting_degree = 3
        if (preliminary_ampl > 0.5) & (preliminary_ampl <=0.7) & (nfas >= (10)):
            starting_degree = 4
        elif (preliminary_ampl > 0.7) & (preliminary_ampl <=0.9) & (nfas >= (12)):
            starting_degree = 5
        elif (preliminary_ampl > 0.9) & (nfas >= (14)):
            starting_degree = 6
    else:
        starting_degree = starting_degree

    #Find the maxgap
    maxgap=max_gap(x)
    if maxgap>maxgap_threshold:
        max_degree = starting_degree + 1

    max_degree = min([max_degree, int(np.floor((nfas-1.)/2.))])
        
    if do_plot_fourfit == True:
        if findblazhko:
            fig = plt.figure(figsize=(8, 11))
            fig.subplots_adjust(top=.99,bottom=.05)
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312, sharex = ax1)
            ax3 = fig.add_subplot(313)
        else:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(211)
            ax3 = fig.add_subplot(212)

        #     ax1.plot(x,y,'o')
        ax1.errorbar(x, y, yerr=err, marker='.', ls='')
        ax1.axhline(np.percentile(y, perc_hi), linestyle='--')
        ax1.axhline(np.percentile(y, perc_lo), linestyle='--')
        ax1.axhline(preliminary_mean + .48 * ampl2, linestyle='--', color='r')
        ax1.axhline(preliminary_mean - .48 * ampl2, linestyle='--', color='r')

        ax3.errorbar([w for z in [x, x + 1] for w in z],
                     [w for z in [y, y] for w in z],
                     yerr=[w for z in [err, err] for w in z], marker='.', ls='')
        ax3.invert_yaxis()

    for degree in starting_degree + np.arange(max_degree + 1 - starting_degree):
        #         print('degree=',degree)
#         print('---degree---')
#         print(degree)
        a_init = [1.0, .1] * degree + [np.mean(y)]
        #         popt, pcov = curve_fit(fourier_series, x, y, a_init, sigma=1./(err**2), absolute_sigma=True)
        popt, pcov, ind_clipped, std = curve_fit_sigmaclip(fourier_series, x, y, a_init,
                                                           sigma=1. / (err ** 2), threshold=threshold_clipping,
                                                           absolute_sigma=True, maxfev=maxfev)

        residuals = (y[ind_clipped] - fourier_series(x[ind_clipped], *popt)) ** 2
        residuals_weighted = residuals / err[ind_clipped] ** 2
        n_clipped = len(x[ind_clipped])
        n_param = len(popt)
        chisqs.append(sum(residuals_weighted) / (n_clipped - n_param))
        popts.append(popt)
        stds.append(std)
        degrees.append(degree)

        #         #COMMENTARE!!!!
        #         yfit = fourier_series(xfit, *popt)
        #         ax1.plot(xfit, yfit, '--', label=degree, c='k')
        #         print(chisq_ratio)

        chisq_ratio = chisqs[degree - starting_degree] / chisqs[degree - starting_degree - 1]
        
#         print('---')
#         print(degree, max_degree)
#         print(chisqs[degree-starting_degree])
#         print(chisqs[degree-starting_degree-1])
#         print(chisq_ratio)

        if degree > starting_degree:
            if ((chisq_ratio <= chisq_ratio_threshold) | (chisq_ratio > 1.)) & (degree < max_degree):
                # Aumenta un grado
                if do_plot_fourfit:
                    ax1.plot(xfit, fourier_series(xfit, *popts[-2]), '--', label=degree - 1, linewidth=.3)
            else:
                # Convergiuto
                x = x[ind_clipped]
                y = y[ind_clipped]
                err = err[ind_clipped]
                residuals = (y - fourier_series(x, *popts[-2])) ** 2
                yfit = fourier_series(xfit, *popts[-2])
                
                if do_plot_fourfit:
                    ax1.errorbar(x, y, yerr=err, marker='.', ls='', c='r')
                    ax1.plot(xfit, yfit, label=degree - 1, linewidth=2, c='k')
                    ax1.plot(xfit, yfit + threshold_clipping * std, '--', c='k')
                    ax1.plot(xfit, yfit - threshold_clipping * std, '--', c='k')

                    ax3.errorbar([w for z in [x, x + 1] for w in z],
                                 [w for z in [y, y] for w in z],
                                 yerr=[w for z in [err, err] for w in z], marker='.', ls='', c='r')
                    ax3.plot(xfit, yfit, label=degree - 1, linewidth=2, c='k')
                    ax3.plot(xfit + 1, yfit, label=degree - 1, linewidth=2, c='k')

                ampl = max(yfit) - min(yfit)
                overall_sigma = (np.sqrt(sum(residuals)) / (n_clipped))
                residuals_reduced = (sum(residuals) / (n_clipped - n_param))
                residuals_reduced2 = (sum(residuals) / (n_clipped - n_param) / ampl)
                fraction_rejected = nfas - len(ind_clipped)

                if do_print_fourfit:
                    data = Table()
                    data['ind'] = np.arange(nfit + 1)
                    data['x'] = np.asarray(xfit)
                    data['y'] = np.asarray(yfit)
#                     print(folderout+subfolder_lcvs+name+'_'+filter+suffix_fileout+'_fourx.dat')
                    ascii.write(data, folderout+subfolder_lcvs+name+'_'+filter+suffix_fileout+'_fourx.dat',
                                formats={'ind': '%4i',
                                         'x': '%6.4f',
                                         'y': '%8.4f'}, overwrite=True)

                break

    if do_plot_fourfit:
        ax1.text(.1, max(y), name + ' ' + filter)
        ax1.invert_yaxis()
        ax1.legend()

    outliers_thresholds = [i for i in 1 + np.arange(10)]
    outliers_thresholds.extend([15, 20, 30])
    median_residuals = np.median(residuals)
    meanmag, _ = meanmag_flux(yfit, np.ones(len(yfit)))
    fraction_outliers = [len(residuals[residuals > i * median_residuals]) / len(residuals) for i in
                         outliers_thresholds]

    if findblazhko:
        ind_sort = np.argsort(x)
        maxres = max(residuals)
        
        if do_plot_fourfit:
            ax2.plot(x[ind_sort], residuals[ind_sort])
            ax2.plot(x, residuals, '.')
            #         fraction_outliers = len(residuals[residuals > threshold_outliers*median_residuals])/len(residuals)

            ax2.axhline(median_residuals)
            ax2.axhline(10 * median_residuals)
            #         ax2.text(.0,.9*maxres,'Outlier fraction: {0:4.2f}'.format(fraction_outliers))
            ax2.text(.0, .9 * maxres, 'Outlier fraction: ' + ' '.join(["%5.3f" % i for i in fraction_outliers]))
            ax2.text(.0, .85 * maxres, 'Chisq: {0:8.3f}'.format(chisqs[-2]))
            ax2.text(.0, .80 * maxres, 'reduced Chisq: {0:8.3f}'.format(chisqs[-2] / ampl))
            ax2.text(.0, .75 * maxres, 'residuals: {0:8.5f}'.format(residuals_reduced))
            ax2.text(.0, .70 * maxres, 'reduced residuals: {0:8.5f}'.format(residuals_reduced2))
            ax2.text(.0, .65 * maxres, 'uniformityKS: {0:8.5f}'.format(uniformityks))
            ax2.text(.0, .60 * maxres, 'uniformityKS rebinned: {0:8.5f}'.format(uniformityks_binned))

            
    if do_plot_fourfit:
        plt.savefig(name+'_'+filter+'.pdf')
        plt.close()

    return {'popt': popts[-2],
            'chisq': chisqs[-2],
            'redchisq': chisqs[-2]/ampl,
            'outliers': fraction_outliers,
            'res': residuals_reduced,
            'redres':residuals_reduced2,
            'ampl': ampl,
            'std': stds[-2],
            'overall_sigma': overall_sigma,
            'meanmag' : meanmag,
            'degree': degrees[-2],
            'uniformityKS': uniformityks,
            'uniformityKS_binned': uniformityks_binned,
            'maxgap': maxgap,
            'fraction_rejected': fraction_rejected,
            'fileout':name+'_'+filter+'.pdf',
            'skewness':skew(y),
            'kurtosis':kurtosis(y),
            'xfit':xfit,
            'yfit':yfit}

def fourfit_simple_old(x, y, err, name, filter, subfolder_lcvs, findblazhko=False,
                   threshold_clipping=10., chisq_ratio_threshold=.98,
                  do_print_fourfit=False, max_degree=10):
    sleep(0.005)
#     print(name,filter)
    chisqs=[]
    popts=[]
    stds=[]
    degrees=[]
    nfit=1000
    xfit=np.linspace(0.,1.,nfit+1)
    
    nfas = len(x)
    
    #prima passata: tolgo i punti superoutlier
    ind_clipped = abs( y - np.median(y) < 1.)
#     print("Prima rejection: ",len(x) - len(ind_clipped[ind_clipped]))
    
    x = x[ind_clipped]
    y = y[ind_clipped]
    err = err[ind_clipped]
    
    #Per come è fatto, l'uniformity potrebbe non essere un indicatore ottimale... quindi binno le fasi
    #in bin di binsize (0.01-0.02-0.05) e calcolo l'uniformity sulle fasi binnate.
    x_binned = binned_statistic(x, x, bins=20, range=(0, 1.))[0]
    good=[not isnan(i) for i in x_binned]
    x_binned = x_binned[good]
    uniformityks_binned = uniformityKS(x_binned)
    uniformityks = uniformityKS(x)
        
    perc_lo = 2
    perc_hi = 98
    
    ampl = np.percentile(y,perc_hi) - np.percentile(y,perc_lo)
    ampl2 = .96 * (max(y) - min(y))
    preliminary_mean = (max(y) + min(y))/2.
    
    preliminary_ampl = ampl
    
#     print(len(np.where((y > perc05) & (y < perc95))[0].tolist())/len(y))
#     fraction_within_90perc = len()
    
    starting_degree = 3
    if (preliminary_ampl > 0.5) & (preliminary_ampl <=0.7):
        starting_degree = 4
    elif (preliminary_ampl > 0.7) & (preliminary_ampl <=0.9):
        starting_degree = 5    
    elif preliminary_ampl > 0.9:
        starting_degree = 6    

    #Find the maxgap
    max_degree = 10
    maxgap=max_gap(x)
    if maxgap>0.2:
        max_degree = starting_degree + 1
        
    if findblazhko:
        fig = plt.figure(figsize=(8, 11))
        fig.subplots_adjust(top=.99,bottom=.05)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex = ax1)
        ax3 = fig.add_subplot(313)
    else:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)

#     ax1.plot(x,y,'o')
    ax1.errorbar(x,y,yerr=err,marker='.',ls='')
    ax1.axhline(np.percentile(y,perc_hi),linestyle = '--')
    ax1.axhline(np.percentile(y,perc_lo),linestyle = '--')
    ax1.axhline(preliminary_mean+.48*ampl2,linestyle = '--',color='r')
    ax1.axhline(preliminary_mean-.48*ampl2,linestyle = '--',color='r')
    
    ax3.errorbar([w for z in [x, x+1] for w in z],
                 [w for z in [y, y] for w in z],
                 yerr=[w for z in [err, err] for w in z],marker='.',ls='')
    ax3.invert_yaxis()
    
    for degree in starting_degree+np.arange(max_degree+1-starting_degree):
#         print('degree=',degree)
        a_init=[1.0,.1] * degree + [np.mean(y)]    
#         popt, pcov = curve_fit(fourier_series, x, y, a_init, sigma=1./(err**2), absolute_sigma=True)
        popt, pcov, ind_clipped, std = curve_fit_sigmaclip(fourier_series, x, y, a_init, 
                                         sigma=1./(err**2), threshold=threshold_clipping, absolute_sigma=True)

        residuals = (y[ind_clipped]-fourier_series(x[ind_clipped], *popt))**2
        residuals_weighted = residuals / err[ind_clipped]**2
        n_clipped = len(x[ind_clipped])
        n_param = len(popt)
        
#         print('---')
#         print(n_clipped)
#         print(n_param)        
        
        chisqs.append( sum(residuals_weighted) / (n_clipped - n_param) )        
        popts.append(popt)
        stds.append(std)
        degrees.append(degree)
        
#         #COMMENTARE!!!!
#         yfit = fourier_series(xfit, *popt)
#         ax1.plot(xfit, yfit, '--', label=degree, c='k')
#         print(chisq_ratio)

#         print('---')
#         print(starting_degree)
#         print(degree)
#         print(chisqs[degree-starting_degree])
#         print(chisqs[degree-starting_degree-1])
        chisq_ratio = chisqs[degree-starting_degree]/chisqs[degree-starting_degree-1]

        if degree > starting_degree:
            if ((chisq_ratio <= chisq_ratio_threshold) | (chisq_ratio > 1.)) & (degree < max_degree):
# Aumenta un grado
                ax1.plot(xfit, fourier_series(xfit, *popts[-2]), '--', label=degree-1, linewidth=.3)
            else:
# Convergiuto
                x = x[ind_clipped]
                y = y[ind_clipped]
                err = err[ind_clipped]
                residuals = (y - fourier_series(x, *popts[-2]))**2
                yfit = fourier_series(xfit, *popts[-2])
            
                ax1.errorbar(x,y,yerr=err,marker='.',ls='',c='r')
                ax1.plot(xfit, yfit, label=degree-1, linewidth=2, c='k')
                ax1.plot(xfit, yfit+threshold_clipping*std, '--', c='k')
                ax1.plot(xfit, yfit-threshold_clipping*std, '--', c='k')

                ax3.errorbar([w for z in [x, x+1] for w in z],
                 [w for z in [y, y] for w in z],
                 yerr=[w for z in [err, err] for w in z],marker='.',ls='',c='r')
                ax3.plot(xfit, yfit, label=degree-1, linewidth=2, c='k')
                ax3.plot(xfit+1, yfit, label=degree-1, linewidth=2, c='k')

                ampl = max(yfit) - min(yfit)
                residuals_reduced = (sum(residuals) /  (n_clipped-n_param) )
                residuals_reduced2 = (sum(residuals) /  (n_clipped-n_param) / ampl )
                fraction_rejected = nfas - len(ind_clipped)
 
                if do_print_fourfit:
                    data = Table()
                    data['ind'] = np.arange(nfit+1)
                    data['x'] = np.asarray(xfit)
                    data['y'] = np.asarray(yfit)
                    ascii.write(data, '../'+subfolder_lcvs+name+'_'+filter+'_fourx.dat', 
                                formats={'ind':'%4i',
                                        'x':'%6.4f',
                                        'y':'%8.4f'}, overwrite=True)  
                
                break

    ax1.text(.1,max(y),name+' '+filter)
    ax1.invert_yaxis()
    ax1.legend()
    
    if findblazhko:
        ind_sort=np.argsort(x)
        ax2.plot(x[ind_sort],residuals[ind_sort])
        ax2.plot(x,residuals,'.')
        median_residuals = np.median(residuals)
#         fraction_outliers = len(residuals[residuals > threshold_outliers*median_residuals])/len(residuals)

        outliers_thresholds = [i for i in 1+np.arange(10)]
        outliers_thresholds.extend([15, 20, 30])
        fraction_outliers = [len(residuals[residuals > i*median_residuals])/len(residuals) for i in outliers_thresholds]
        meanmag,_ = meanmag_flux(yfit,np.ones(len(yfit)))
        
        ax2.axhline(median_residuals)
        ax2.axhline(10*median_residuals)
        maxres = max(residuals)
#         ax2.text(.0,.9*maxres,'Outlier fraction: {0:4.2f}'.format(fraction_outliers))
        ax2.text(.0,.9*maxres,'Outlier fraction: '+' '.join(["%5.3f" % i for i in fraction_outliers]))
        ax2.text(.0,.85*maxres,'Chisq: {0:8.3f}'.format(chisqs[-2]))
        ax2.text(.0,.80*maxres,'reduced Chisq: {0:8.3f}'.format(chisqs[-2]/ampl))
        ax2.text(.0,.75*maxres,'residuals: {0:8.5f}'.format(residuals_reduced))
        ax2.text(.0,.70*maxres,'reduced residuals: {0:8.5f}'.format(residuals_reduced2))
        ax2.text(.0,.65*maxres,'uniformityKS: {0:8.5f}'.format(uniformityks))
        ax2.text(.0,.60*maxres,'uniformityKS rebinned: {0:8.5f}'.format(uniformityks_binned))
    
    plt.savefig(name+'_'+filter+'.pdf')
    plt.close()
        
    return {'popt': popts[-2], 
            'chisq': chisqs[-2], 
            'redchisq': chisqs[-2]/ampl, 
            'outliers': fraction_outliers,
            'res': residuals_reduced, 
            'redres':residuals_reduced2,
            'ampl': ampl,
            'std': stds[-2],
            'meanmag' : meanmag,
            'degree': degrees[-2], 
            'uniformityKS': uniformityks,
            'uniformityKS_binned': uniformityks_binned,
            'maxgap': maxgap,
            'fraction_rejected': fraction_rejected,
            'fileout':name+'_'+filter+'.pdf',
            'skewness':skew(y),
            'kurtosis':kurtosis(y),
            'xfit':xfit,
            'yfit':yfit}

def uniformityKS(phases):
    sleep(0.005)
    if(len(phases))>1:
        phase_sort=np.sort(phases)
        n_cum = np.arange(1, len(phases) + 1) / float(len(phases))
        D_max = np.max(np.abs(n_cum - phase_sort - phase_sort[0]))# ma in origine era phase_u_sort[1] ma non capisco il perché
    else:
        D_max=999.
    return D_max