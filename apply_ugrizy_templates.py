import numpy as np
import pandas as pd
from scipy import optimize
from mpfit import mpfit
import sys
from magmed_flux import magmed
import matplotlib.pyplot as plt
from matplotlib import cm

filter_dict_ZTF = {2:'zg', 3:'zr', 4:'zi'}
inverted_filter_dict_ZTF = {v: k for k, v in filter_dict_ZTF.items()}
filter_dict={2:'g', 3:'r', 4:'i'}
templatebin_dict = {0:'RRc(g) 1', 1:'RRc(g) 2', 2:'RRc(g) 3', 
                        3:'RRc(r) 1', 4:'RRc(r) 2', 5:'RRc(r) 3', 
                        6:'RRc(i) 1', 7:'RRc(i) 2', 8:'RRc(i) 3', 
                        9:'RRab(g) 1', 10:'RRab(g) 2', 11:'RRab(g) 3', 
                        12:'RRab(g) 4', 13:'RRab(g) 5', 14:'RRab(g) 6', 
                        15:'RRab(g) 7', 16:'RRab(g) 8', 17:'RRab(g) 9', 
                        18:'RRab(r) 1', 19:'RRab(r) 2', 20:'RRab(r) 3', 
                        21:'RRab(r) 4', 22:'RRab(r) 5', 23:'RRab(r) 6', 
                        24:'RRab(r) 7', 25:'RRab(r) 8', 26:'RRab(r) 9', 
                        27:'RRab(i) 1', 28:'RRab(i) 2', 29:'RRab(i) 3', 
                        30:'RRab(i) 4', 31:'RRab(i) 5', 32:'RRab(i) 6', 
                        33:'RRab(z) 1', 34:'RRab(z) 2', 35:'RRab(z) 3', 
                        36:'RRab(z) 4'}

def weighted_avg_and_std(values, weights):
    """
    Returns the weighted average and standard deviation.

    :param values: values for which the weighted average should be computed (numpy.ndarray)
    :param weights: weights relative to values. (numpy.ndarray)
    :return y: weighted average and variance (set)
    """
    values=np.asarray(values)
    weights=np.asarray(weights)
    
    average = np.average(values, weights=weights)
    sumw = np.sum(weights)
    
    rescale_coeff = 1.
    if len(values) > 1:
        rescale_coeff = len(values)/(len(values)-1)
    
#     variance = np.average((values-average)**2, weights=weights)
    variance_bevington = rescale_coeff*np.sum(weights*(values-average)**2)/sumw
    
    err = np.sqrt(1/sumw)
    err_variance = np.sqrt(variance_bevington + err**2)
    
    return (average, err_variance, err)

def load_coefficient_table(filein):
    """
    Returns the a pandas.DataFrame object containing the coefficients of the 
    analytical forms of the templates.

    :param filein: string with the entire path to the coefficients.csv file (str).
    :return df_coeff: table of the coefficients of the templates (pandas.DataFrame)
    """

    df_coeff = pd.read_csv(filein)
    return df_coeff

def fourier_series(x, *a):
    degree = (len(a)-1)/2    
    ret = a[-1] # the constant is at the last place of the coefficients list
    for deg in range(int(degree)):
        ret += a[deg*2] * np.cos((deg+1)*2*np.pi * x + a[1 + deg*2])
    return ret

def four_for_templatefit(x, Delta_phase, Delta_mag, ampl, templatebin_int, 
                         filein='/home/vittorioinaf/Documenti/Science/RRLyr/Template_ugrizy/Template_analytics/templates_analytics_230901.csv'):

    '''
    gaupe function calculates the value of a Fourier series function up to the
    40th order

    :param x: phases at which the Fourier function should be calculated (array or list)
    :return y: values of the Fourier function, with c coefficients, at phases x (list)
    '''
    
    # Read the coefficients table
    coeff = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin
    coeff = coeff[coeff['index'] == templatebin_int]

    coeff.drop(columns=['col1', 'index', 'degree', 
       'templatebin', 'bin_threshold_lo', 'bin_threshold_hi', 'filter',
       'n_rrls_in_bin', 'n_phasepoints_in_bin', 'npixels_x', 'npixels_y',
       'n_bins_to_average_for_density', 'pulsation_type', 'overall_sigma'], inplace=True)

    Delta_phase = float(Delta_phase)
    Delta_mag = float(Delta_mag)
    ampl = float(ampl)
    
    coeff = coeff.loc[:].values.flatten().tolist()
    new_list = []
    for item in coeff:
        if str(item) != 'nan':
            new_list.append(item)
    coeff = new_list

    y = fourier_series(x + Delta_phase, *coeff)
    y = Delta_mag + np.asarray(y) * ampl

    return y

def myfunct_four_for_templatefit(p, fjac=None, x=None, y=None, err=None):
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    
    model = four_for_templatefit(x, p[0], p[1], p[2], p[3])
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    
    return [status, (y-model)/err]

def find_templatebin(pulsation_type, period, passband):

    '''
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :param passband: passband (2,3,4 for g,r,i) (np.float64)
    :return templatebin: index to search in the coefficient table (int)
    '''

    filein = '/home/vittorioinaf/Documenti/Science/RRLyr/Template_ugrizy/Template_analytics/templates_analytics_230901.csv'
    df = load_coefficient_table(filein)
    
    df = df[(df['filter'] == passband) & (df['pulsation_type'] == pulsation_type) & \
        (df['bin_threshold_lo'] <= period) & (df['bin_threshold_hi'] > period)]
    
    templatebin = df['index'].values
    
    return(templatebin)

def anchor_template(HJD, mag, err, pulsation_type,
                    period, t0, passband, ampl, 
                    filein, figure_out='', quiet=1):

    '''
    anchor_template function anchors the right template (selected
    by means of the parameters pulsation_type, period and passband)
    on a series of magnitude measurements

    :param HJD: list of Heliocentric Julian Dates for the RV measurements (list)
    :param mag: list of mag measurements (list)
    :param err: list of uncertainties on mags (list)
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :param t0: Anchor epoch in HJD (np.float64)
    :param passband: Passband (int)
     Possible values: 2, 3, 4, 5 for g, r, i, z, respectively
    :param ampl: amplitude in the selected passband (np.float64)
    :param filein: path to the coefficients table in csv format (string)
    :param figure_out: path to the output figure. '' if no outpmean_mag_mean = magmed(mean_mag_list, err2=mean_mag_err_list)ut figure is desired. (string)
    :param quiet: 0/1 to allow/forbid mpfit to print the results of the iterations (int)
    :return: data_return: dictionary including the following entries:
     'v_gamma_list': list of systemic velocities from each RV measurement
     'xfit': grid of phases
     'yfit_list': template fit values for each RV measuremnet (list of lists)
     'v_gamma_mean': 2-element tuple including average and standard deviation of the systemic velocity
     (dictionary)
    '''

    templatebin_int = find_templatebin(pulsation_type, period, passband)[0]
    
    # Derive the phase from HJD, t0 and period
    phase = (HJD-t0)/period % 1.
    n_points = len(phase)

    mean_mag_guess = np.mean(mag)
    
    # Generates the grid of phases to evaluate the template
    n_phases_for_model = 1000
    xfit = (np.arange(n_phases_for_model + 1) / float(n_phases_for_model))

    # Read the coefficients table
    c = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin    
    c = c[c['index'].values == templatebin_int]
    
    # Estimate mean magnitude for each magnitude measurement
    mean_mag_list = []
    mean_mag_err_list = []
    yfit_list = []
    
    deltaphase_guess = 0
    
    for phase_i, mag_i, err_i in zip(phase, mag, err):

        deltamag = mag_i - four_for_templatefit(phase_i, 0., 0, ampl, templatebin_int)
        yfit = four_for_templatefit(xfit, 0., deltamag, ampl, templatebin_int)

        # Calculates the mean magnitude
        mean_mag_temp = magmed(yfit)
        mean_mag_err_temp = np.sqrt(err_i**2 + (ampl*c.overall_sigma.values[0])**2)

        mean_mag_list.append(mean_mag_temp)
        mean_mag_err_list.append(mean_mag_err_temp)
        yfit_list.append(yfit)

    if n_points > 1:
        mean_mag_list = np.asarray(mean_mag_list)
        mean_mag_mean_array = magmed(mean_mag_list, err2=mean_mag_err_list)
        mean_mag_mean = mean_mag_mean_array[0]
        err_mag_mean = mean_mag_mean_array[1]
    else:
        mean_mag_mean = mean_mag_list[0]
        err_mag_mean = mean_mag_err_list[0]
        
    if figure_out != '':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        cmap_size = 256
        cmap = cm.get_cmap('Oranges', cmap_size)
        cmap_step = np.ceil(cmap_size/len(phase))
        cmap_zero = np.floor(cmap_step/2)

        iii = 0
        for phase_i, mag_i, err_i in zip(phase, mag, err):

            ax.plot(xfit, yfit_list[iii], c=cmap(int(cmap_zero + iii*cmap_step)), zorder=0)
            ax.plot([-1, 2], [mean_mag_list[iii], mean_mag_list[iii]], '--',
                    c=cmap(int(cmap_zero + iii*cmap_step)), zorder=0)
            ax.errorbar(phase_i, mag_i, yerr=err_i, c=cmap(int(cmap_zero + iii*cmap_step)), zorder=1)
            ax.scatter(phase_i, mag_i, c=[cmap(int(cmap_zero + iii*cmap_step))], s=20, zorder=1)
            iii = iii + 1

        ax.plot([-1,2], [mean_mag_mean, mean_mag_mean], c='k', zorder=2)
        ax.set_xlim([0.,1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('PHASE', fontsize=18)
        plt.ylabel('mag [mag]', fontsize=18)
        fig.savefig(figure_out)
        plt.close()

    return {'mean_mag_list': mean_mag_list, 'xfit': xfit, 'yfit_list': yfit_list,
            'mean_mag_mean': mean_mag_mean, 
            'mean_mag_err': np.sqrt(err_mag_mean**2 + (ampl*c.overall_sigma.values[0])**2)}
        
def apply_templatefit(HJD, mag, err, pulsation_type,
                            period, t0, passband,
                            filein, ampl=.6, figure_out='', quiet=1, 
                            free_amplitude=True, amplmax=2, amplmin=.1):

    '''
    apply_template_threepoints function applies the right template (selected
    by means of the parameters pulsation_type, period and passband)
    on a series of magnitude measurements

    :param HJD: list of Heliocentric Julian Dates for the RV measurements (list)
    :param mag: list of mag measurements (list)
    :param err: list of uncertainties on mags (list)
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :param t0: Anchor epoch in HJD (np.float64)
    :param passband: Passband (int)
     Possible values: 2, 3, 4, 5 for g, r, i, z, respectively
    :param ampl: amplitude in the selected passband (np.float64)
    :param free_amplitude: should the amplitude be optimized in the fit or fixed? (bool)
    :param filein: path to the coefficients table in csv format (string)
    :param figure_out: path to the output figure. '' if no output figure is desired. (string)
    :param quiet: 0/1 to allow/forbid mpfit to print the results of the iterations (int)
    :return: data_return: dictionary including the following entries:
     'v_gamma_list': list of systemic velocities from each RV measurement
     'xfit': grid of phases
     'yfit_list': template fit values for each RV measuremnet (list of lists)
     'v_gamma_mean': 2-element tuple including average and standard deviation of the systemic velocity
     (dictionary)
    '''

    templatebin_int = find_templatebin(pulsation_type, period, passband)[0]
    
    # Derive the phase from HJD, t0 and period
    phase = (HJD-t0)/period % 1.

    mean_mag_guess = np.mean(mag)

    #     ARV = amplitude_rescale(AV, pulsation_type, diagnostic_int)
    if free_amplitude:
        ampl_fixed = 0
    else:
        ampl_fixed = 1
    
    # Generates the grid of phases to evaluate the template
    n_phases_for_model = 1000
    xfit = (np.arange(n_phases_for_model + 1) / float(n_phases_for_model))

    # Read the coefficients table
    c = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin    
    c = c[c['index'].values == templatebin_int]

    # Generates the first guess on the coefficients
    n_guesses = 3
    deltaphase_guesses = np.arange(n_guesses)/float(n_guesses)

    chisqs=[]
    popts=[]
    model_points=[]
    for deltaphase_guess in deltaphase_guesses:
        
        p0 = (deltaphase_guess, mean_mag_guess, ampl, templatebin_int)

        parinfo = [{'value': deltaphase_guess, 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': mean_mag_guess, 'fixed': 0, 'limited': [1, 1], 
                    'limits': [mean_mag_guess - amplmax/1.8, mean_mag_guess + amplmax/1.8]},
                   {'value': ampl, 'fixed': ampl_fixed, 'limited': [1, 1], 'limits': [amplmin, amplmax]},
                   {'value': templatebin_int, 'fixed': 1}]

        fa = {'x': phase, 'y': mag, 'err': err}
        m = mpfit(myfunct_four_for_templatefit, p0, parinfo=parinfo, functkw=fa, quiet=quiet)
        
        model = four_for_templatefit(phase, m.params[0], m.params[1], m.params[2], m.params[3])
        chisq = (myfunct_four_for_templatefit(m.params, x=phase, y=mag, err=err)[1] ** 2).sum()
        
        chisqs.append(chisq)
        model_points.append(model)
        popts.append(m.params)

    chisqs = np.asarray(chisqs)
    ind_best = chisqs.argmin()
    
    yfit = four_for_templatefit(xfit, *popts[ind_best], filein)
    mag_mean = np.mean(yfit)
    errmag_mean = np.sqrt(np.diag(m.covar))[1]

    if figure_out != '':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        ax.plot(xfit, yfit, 'k', zorder=0)
        ax.plot([-1, 2], [mag_mean, mag_mean], '--', c='k', zorder=0)
        ax.errorbar(phase, mag, yerr=err, c='r', fmt = ' ', zorder=1)
        ax.scatter(phase, mag, c='r', s=20, zorder=1)

        ax.set_xlim([0.,1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('PHASE', fontsize=18)
        plt.ylabel(filter_dict[passband]+' [mag]', fontsize=18)
        fig.savefig(figure_out)
        plt.close()

    return {'xfit': xfit, 'yfit': yfit,
            'mag_mean': mag_mean, 
            'errmag_mean': np.sqrt(errmag_mean**2 + (ampl*c.overall_sigma.values[0])**2),
            'popts': popts[ind_best], 'chisq':chisqs[ind_best], 'model_points': model_points[ind_best]}