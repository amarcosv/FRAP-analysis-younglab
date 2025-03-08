import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from skimage.morphology import disk, remove_small_holes, erosion
from skimage.filters import rank, threshold_otsu, gaussian , threshold_triangle, threshold_mean
from skimage.measure import label
from scipy.stats import chi2
from scipy.stats import chisquare

AREA_THRESHOLD = 5000


def process_ROI(imageData, frap_experiment, regionsInfo, frameInfo, do_bkg=False):

    roiData = pd.DataFrame(columns = ['timepoint','timepoint_frap','timestamp','timestamp_frap', 'bleach', 'control', 'bkg'])
    roiData['timepoint']=range(imageData.shape[0])
    #print(frap_experiment) 
    roiData['timepoint_frap']=roiData['timepoint']-frap_experiment.bleach_frame.item()
    roiData['bkg'] = np.zeros(imageData.shape[0])
    roiData['timestamp'] = frameInfo['Time[s]']
    roiData['timestamp_frap']=roiData['timestamp']-roiData.loc[regionsInfo.loc[0,'bleach_frame'] ,'timestamp']
    frap_experiment['dt'] = np.mean(np.diff(roiData['timestamp'] [frap_experiment.bleach_frame.item()::]))
    frap_experiment['nframes'] = len(frameInfo['Time[s]']) 
    roiData['stage_x'] = frameInfo['X[micron]']
    roiData['stage_y'] = frameInfo['Y[micron]']
    roiData['stage_z'] = frameInfo['Z[micron]']

    roi_center = [0,0]
    #Identify regions using metadata
    for i,region in regionsInfo.iterrows():
        
        roi = imageData[:, int(regionsInfo.loc[i,'Y_roi']) : int(regionsInfo.loc[i,'Y_roi'] + regionsInfo.loc[i,'Height_roi']),
               int(regionsInfo.loc[i,'X_roi']) : int(regionsInfo.loc[i,'X_roi'] + regionsInfo.loc[i,'Width_roi'])]

        
        roiMean = np.mean(roi, axis = (1,2))

        if region['IsForBleach']:
            roiData.loc[0:roiMean.size, ['bleach']] = roiMean
            roi_center = [ int(regionsInfo.loc[i,'Y_roi'] + regionsInfo.loc[i,'Height_roi']/2),
               int(regionsInfo.loc[i,'X_roi'] + (regionsInfo.loc[i,'Width_roi']/2))]           

        else: 
            roiData.loc[0:roiMean.size, ['control_roi']] = roiMean

    #Perform cell segmentation and extract
    if frap_experiment.wcell_corr.item():
        wcellMask = find_wcell_roi(imageData[0:frap_experiment.bleach_frame.item()-1,:,:],roi_center)
        #wcellMask = find_wcell_roi(imageData[-10:,:,:],roi_center)
        wcellMean = np.zeros(imageData.shape[0])
        for t in range(imageData.shape[0]):
            wcellMean[t] = np.mean(imageData[t,wcellMask])  

        roiData['control_wcell'] = wcellMean
        frap_experiment['wcellMask'] = [wcellMask]


    if do_bkg:
        p5 = np.percentile(imageData, 5)
    

    return roiData, frap_experiment

# segment cell shape from a stack of frames
def find_wcell_roi(image, roi_center):
    
    Tmean = np.mean(image, axis = 0, dtype = np.uint16)

    Tmean_f = rank.mean(Tmean, footprint = disk(3))
    Tmean_f = gaussian(Tmean, sigma=3)

    th = threshold_otsu(Tmean_f)
    mask =Tmean_f>th

    mask  = remove_small_holes(mask, area_threshold = AREA_THRESHOLD)

    mask = erosion(mask, disk(5))

    label_image = label(mask)
    cellID = label_image[roi_center[0],roi_center[1]]

    # If there is a brighter cell and we dont detect it, try again with other threshold
    if cellID == 0:
        #Tmean_f[cellID>0] = Tmean_f[cellID>0]-th*0.5
        th = threshold_mean(Tmean_f)
        mask =Tmean_f>th

        mask  = remove_small_holes(mask, area_threshold = AREA_THRESHOLD)

        mask = erosion(mask, disk(5))

        label_image = label(mask)
        cellID = label_image[roi_center[0],roi_center[1]]

    mask = label_image ==cellID

    return mask
# Double exponential model for photobleaching decay
def double_exponential(t, const, amp_slow, amp_fast, tau_fast, tau_multiplier):
    '''Compute a double exponential function with constant offset. 
    Function contains a slow and a fast component in a linear combination. Fast component decay is expressed as times the slow component
    Parameters:
    t       : Timestamp vector in seconds.
    const   : Amplitude of the constant offset. 
    amp_fast: Amplitude of the fast component.  
    amp_slow: Amplitude of the slow component.  
    tau_slow: Time constant of slow component in seconds.
    tau_factor: Time constant of fast component relative to slow. 
    '''
    tau_slow = tau_fast*tau_multiplier
    return const + amp_slow*np.exp(-t*tau_slow) + amp_fast*np.exp(-t*tau_fast) 

# Single exponential model for photobleaching decay
def single_exponential(t, const, amp, tau):
    '''Compute a double exponential function with constant offset. 
    Function contains a slow and a fast component in a linear combination. Fast component decay is expressed as times the slow component
    Parameters:
    t       : Timestamp vector in seconds.
    const   : Amplitude of the constant offset. 
    amp     : Amplitude of the exponential function
    tau     : Time constant  in seconds.
    '''
    
    return const + amp*np.exp(-tau*t) 

# Calculate R-square value of the fit
def calculate_fit_qc(data, fit_data, nparams, sigma):
    
    
    #Chi squared p-value
    degrees_of_freedom = len(data) - nparams

    sigma = np.std((data-fit_data)) 

    chisq = np.sum(((data-fit_data)/sigma)**2)

    r_chisq = chisq/degrees_of_freedom
    p_value = 1- chi2.cdf(chisq, degrees_of_freedom)

    #chisq, p_value = chisquare(data, f_exp =  fit_data, ddof=nparams, sum_check= True)

    
    # residual sum of squares
    ss_res = np.sum((data - fit_data) ** 2)

    # total sum of squares
    ss_tot = np.sum((data - np.mean(data)) ** 2)

    # r-squared 
    r2 = 1 - (ss_res / ss_tot)

    return r2, r_chisq, p_value

#Find curve parameters for a decaying exponential curve
def fit_photobleaching_exp(time, data, order):

    max_sig = np.max(data)

    
    if order==1:
        inital_params = [max_sig/2, max_sig/4, 1]
        bounds = ([0      , 0      , 0  ],
             [ max_sig, max_sig, 1000])
        photobleach_decay_params, parm_cov = curve_fit(single_exponential, time, data, 
                                  p0=inital_params, bounds=bounds, maxfev=1000)
        #photobleach_decay = single_exponential(roiData['timestamp_frap'], *photobleach_decay_params)
        #r_squared = calculate_r_squared(data,  single_exponential(time,*photobleach_decay_params))

    else:
        inital_params = [max_sig/2, max_sig/4, max_sig/4, 1, 0.1]
        bounds = ([0      , 0      ,0   , 0 ,   0  ],
                [ max_sig, max_sig, max_sig, 100,1])
        photobleach_decay_params, parm_cov = curve_fit(double_exponential, time, data, 
                                  p0=inital_params, bounds=bounds, maxfev=1000)
        #photobleach_decay = double_exponential(roiData['timestamp_frap'], *photobleach_decay_params)
        #r_squared = calculate_r_squared(data,  double_exponential(time,*photobleach_decay_params))

    return photobleach_decay_params

# Estimate curve Y values based on timepoints and exponential order
def estimate_exp_curve(time, params, order):
    
    if order==1:
        
        exp_curve_values = single_exponential(time, *params)       

    else:
       
        exp_curve_values = double_exponential(time, *params)
        
    return exp_curve_values



# Do photobleaching correction using a mono or bi-exponential model
def photobleaching_corr(roiData, ref_roi, frap_experiment, delay=3, exp=1):

    print('[photobleaching_corr] Calculating photobleaching from imaging using ' + ref_roi + ' reference region')
    print('\tusing data from frame ' + str(frap_experiment.bleach_frame.item() + delay) + ' onwards (' + str(delay) + ' after roi bleaching)')

    #Save the data used for bleaching corr as 'reference'
    roiData['reference'] = roiData[ref_roi].to_numpy()
   
   
    # Use data from bleach timepoint + delay to avoid including in the 
    # fitting timpoints showing a dip from recovery from photobleaching in the reference area
    reference_data = roiData[ref_roi].iloc[(frap_experiment.bleach_frame.item() + delay)::].to_numpy()
    time_data = roiData['timestamp_frap'].iloc[(frap_experiment.bleach_frame.item() + delay)::].to_numpy()

    sigma = np.std(roiData[ref_roi].iloc[0:(frap_experiment.bleach_frame.item()-1)].to_numpy())

    sigma = np.std(reference_data[-10:])
    
    # Initial guess for parameters
    y_o = np.mean(reference_data[-10:])
    A_o = np.mean(roiData[ref_roi].iloc[0:frap_experiment.bleach_frame.item()-1].to_numpy()) - y_o
    tau_o = 0.05

    bounds = ([y_o/2, A_o-np.abs(A_o/2), 0 ],
             [ y_o*2, A_o+np.abs(A_o/2), 10])
    print([y_o, A_o, tau_o])
    print(bounds)
    #max_sig = np.max(reference_data)
    #bounds = ([0      , 0      , 0  ],
    #         [ max_sig, max_sig, 1000])
   
    # Fit monoexponential decay curve
    photobleach_decay_params, parm_cov = curve_fit(single_exponential, time_data, reference_data, 
                                  p0=[y_o, A_o, tau_o], bounds = bounds, sigma = sigma, absolute_sigma=True)


    #Find fitting parameters for exponential function
    #photobleach_decay_params = fit_photobleaching_exp(time_data, reference_data, exp)

    #Extrapolate  decay curve for entire experiment
    photobleach_decay = estimate_exp_curve(roiData['timestamp_frap'], photobleach_decay_params, exp)

    #Calculate fitting error r_squared and chi_sq just for the timepoints used in the curve fitting step
    [r_squared, chi_squared, p_val] = calculate_fit_qc(reference_data,  estimate_exp_curve(time_data, photobleach_decay_params, exp), len(photobleach_decay_params), sigma)
    
    print('Fit results:')    

    if exp==1:
        print('\toffset = '+ str(photobleach_decay_params[0]) + 
          '\n\tamplitude: ' +  str(photobleach_decay_params[1]) + 
          '\n\ttau: ' + str(photobleach_decay_params[2])  )
    else:
        print('\toffset = '+ str(photobleach_decay_params[0]) + 
          '\n\tamplitude_slow: ' +  str(photobleach_decay_params[1]) +
          '\n\tamplitude_fast: ' +  str(photobleach_decay_params[2]) +  
          '\n\ttau_slow: ' + str(photobleach_decay_params[3]) + 
          '\n\ttau_fast: ' + str(photobleach_decay_params[4] * photobleach_decay_params[3])) 
      
    print('\tFitting error (r2)= ' + str(r_squared))
    print('\tFitting goodness (chi2)= ' + str(chi_squared))
    print('\tFitting significance (p-value)= ' + str(p_val))

    #Extrapolated fitted decay curve
    roiData['reference_decay_curve'] = photobleach_decay
    #Fitted decay curve with original prebleach data
    roiData['reference_synth'] = photobleach_decay
    roiData.loc[0:(frap_experiment.bleach_frame.item()-1),'reference_synth'] = roiData['reference'].iloc[0:(frap_experiment.bleach_frame.item())]
    
    #Correct reference and bleach region data with extrapolated fitted curve normalized to bleach time point
    roiData['bleach_photo_corr'] = roiData['bleach'] / (photobleach_decay/photobleach_decay[frap_experiment.bleach_frame.item() ])
    roiData['reference_photo_corr'] =  roiData['reference'] / (photobleach_decay/photobleach_decay[frap_experiment.bleach_frame.item() ])

    frap_experiment['photobleach_fit'] = [photobleach_decay_params]
    frap_experiment['photobleach_fit_r2'] = r_squared
    frap_experiment['photobleach_fit_chi2'] = chi_squared
    frap_experiment['photobleach_fit_pval'] = p_val
    

    frap_experiment

    return roiData , frap_experiment    

#Peform Double normalization
def run_double_normalization(roiData, frap_experiment):
    '''Normalized curves using a single or double normalization approach     
    Calculated parameters:
    pre-reference       : Average of reference for all frames before bleaching
    pre-reference       : Value of fitted reference after bleaching (avoids dip)
    pre-bleach          : Average of bleach roi for all frames before bleaching
    post-bleach         : Value of bleach roi after bleaching
    ref_norm            : Fitted curve of reference intensity normalized to intensity at bleaching event
    ref_norm_raw        : Raw reference intensity normalized to average of frames before bleaching
    frap_norm           : Double normalized curve
    frap_fullscale_norm : Full scale normalized curve (bleach timepoint = 0)
    gap_ratio           : For w_cell reference: Use avg 10 frames post bleach / avg pre bleach
                        : for roi reference: Use extrapolated pre intensity / measured pre intensity    
    bleach_depth        : intensity from double normalized frap curve at bleach time (dip in 0-1 scale)
    '''
    frap_experiment['pre-reference'] = np.mean(roiData['reference_synth'].iloc[0:frap_experiment.bleach_frame.item()-1])
    frap_experiment['post-reference'] = np.mean(roiData['reference_synth'].iloc[frap_experiment.bleach_frame.item()])
    frap_experiment['pre-bleach'] = np.mean(roiData['bleach'].iloc[0:frap_experiment.bleach_frame.item()-1])
    frap_experiment['post_bleach'] = roiData['bleach'].iloc[frap_experiment.bleach_frame.item()]
    
    #Single normalization
    roiData['ref_norm'] = (roiData['reference_synth'] / roiData['reference_synth'].iloc[frap_experiment.bleach_frame.item()])
    roiData['ref_norm_raw'] = roiData['reference'] / np.mean(roiData['reference'].iloc[0:frap_experiment.bleach_frame.item()-1])
    
    #Double normalized curve (Prebleach set to 1)
    roiData['frap_norm'] = (frap_experiment['pre-reference'].iloc[0]/roiData['reference_synth'] ) \
         * (roiData['bleach'] /frap_experiment['pre-bleach'].iloc[0])   

    #Full scale normalization (only use for diffusion coefficient calculation)
    roiData['frap_fullscale_norm'] = (roiData['frap_norm'] - roiData['frap_norm'].iloc[frap_experiment.bleach_frame.item()]) / \
    (np.mean(roiData['frap_norm'].iloc[0:frap_experiment.bleach_frame.item()-1])  - roiData['frap_norm'].iloc[frap_experiment.bleach_frame.item()] ) 
    
    #Gap ratio calculation depends on choice of reference region
    if frap_experiment.wcell_corr.item():        
        frap_experiment['gap_ratio'] = np.mean(roiData['reference_synth'].iloc[frap_experiment.bleach_frame.item():frap_experiment.bleach_frame.item()+10])  \
        / np.mean(roiData['reference'].iloc[0:frap_experiment.bleach_frame.item()-1])
             
    else:
        frap_experiment['gap_ratio'] =  np.mean(roiData['reference_decay_curve'].iloc[0:frap_experiment.bleach_frame.item()-1]) \
            / np.mean(roiData['reference'].iloc[0:frap_experiment.bleach_frame.item()-1])
    

    frap_experiment['bleach_depth'] = roiData['frap_norm'].iloc[frap_experiment.bleach_frame.item()] 

    return roiData, frap_experiment   



def pre_bleach_normalization(roiData, frap_experiment):

    frap_experiment['pre-bleach_reference_corr'] = np.mean(roiData['reference_photo_corr'].iloc[0:frap_experiment.bleach_frame.item()-1])
    frap_experiment['pre-bleach_reference'] = np.mean(roiData['reference'].iloc[0:frap_experiment.bleach_frame.item()-1])
    frap_experiment['post-bleach_reference_corr'] = np.mean(roiData['reference_photo_corr'].iloc[frap_experiment.bleach_frame.item():frap_experiment.bleach_frame.item()+10])
    frap_experiment['pre-bleach_bleach'] = np.mean(roiData['bleach_photo_corr'].iloc[0:frap_experiment.bleach_frame.item()-1])

    frap_experiment['gap_ratio'] = frap_experiment['pre-bleach_reference'] / frap_experiment['pre-bleach_reference_corr'] 
    frap_experiment['gap_ratio'] = frap_experiment['post-bleach_reference_corr'] / frap_experiment['pre-bleach_reference_corr'] 

    roiData['reference_photo_corr_norm'] = roiData['reference_photo_corr'] / frap_experiment['pre-bleach_reference'].item()      
    roiData['bleach_photo_corr_norm'] = roiData['bleach_photo_corr'] / frap_experiment['pre-bleach_bleach'].item() 
    frap_experiment['bleach_depth'] = roiData['bleach_photo_corr_norm'].iloc[frap_experiment.bleach_frame.item()]    

    return roiData, frap_experiment   


# Fit recovery curve using a mono or bi-exponential model
def fit_recovery_curve(roiData, frap_experiment, exp=1):
    print('Fitting recovery model')

    
    
    #bleach_data = roiData['bleach_photo_corr_norm'].iloc[frap_experiment.bleach_frame.item()::]
    bleach_data = roiData['frap_norm'].iloc[frap_experiment.bleach_frame.item()::].astype(float).to_numpy()
    time_data = roiData['timestamp_frap'].iloc[frap_experiment.bleach_frame.item()::].to_numpy()
  
   
    max_sig = np.max(bleach_data)
    sigma = np.std(roiData['frap_norm'].iloc[0:frap_experiment.bleach_frame.item()-1].to_numpy())

    # Initial guess for parameters
    y_o = 1
    A_o = -np.mean(bleach_data[0]) 
    tau_o = 2


    if exp==1:
        inital_params = [y_o, A_o, tau_o]
        bounds = ([y_o/2, A_o/2, 0  ],
             [ y_o*1.5,   A_o*2, 20])    
        bleach_recovery_params, parm_cov = curve_fit(single_exponential, time_data, bleach_data, 
                                  p0=inital_params, maxfev=10000, sigma = sigma, absolute_sigma=True)
        bleach_recovery = single_exponential(time_data, *bleach_recovery_params)

        [r_squared, chi_squared, p_val] = calculate_fit_qc(bleach_data,  single_exponential(time_data,*bleach_recovery_params), len(bleach_recovery_params), sigma)
        
   
    else:
        inital_params = [y_o, A_o, A_o/2, tau_o, 0.1]
        bleach_recovery_params, parm_cov = curve_fit(double_exponential, time_data, bleach_data, 
                                  p0=inital_params,  sigma= sigma, absolute_sigma=True)
        bleach_recovery = double_exponential(time_data, *bleach_recovery_params)
        
        [r_squared, chi_squared, p_val] = calculate_fit_qc(bleach_data,  double_exponential(time_data,*bleach_recovery_params), len(bleach_recovery_params), sigma)


    print('Fit results:')    

    if exp==1:
        print('\toffset = '+ str(bleach_recovery_params[0]) + 
          '\n\tamplitude: ' +  str(bleach_recovery_params[1]) + 
          '\n\ttau: ' + str(bleach_recovery_params[2])  )
    else:
        print('\toffset = '+ str(bleach_recovery_params[0]) + 
          '\n\tamplitude_slow: ' +  str(bleach_recovery_params[1]) +
          '\n\tamplitude_fast: ' +  str(bleach_recovery_params[2]) +  
          '\n\ttau_slow: ' + str(bleach_recovery_params[3]) + 
          '\n\ttau_fast: ' + str(bleach_recovery_params[4] * bleach_recovery_params[3])) 
    
    print('\tFitting error (r2)= ' + str(r_squared))
    print('\tFitting goodness (chi2)= ' + str(chi_squared))
    print('\tFitting significance (p-value)= ' + str(p_val))

    roiData.loc[frap_experiment.bleach_frame.item()::,'bleach_recovery_curve'] = bleach_recovery
    roiData.loc[0:frap_experiment.bleach_frame.item()-1,'bleach_recovery_curve'] = 1

    frap_experiment['recovery_model'] = exp
    frap_experiment['recovery_fit'] = [bleach_recovery_params]
    frap_experiment['recovery_fit_r2'] = r_squared
    frap_experiment['recovery_fit_chi2'] = chi_squared
    frap_experiment['recovery_fit_pval'] = p_val

    #frap_experiment['mob'] = -bleach_recovery_params[1] / (1-(bleach_recovery_params[0] + bleach_recovery_params[1] ))  
    if exp==1:
        frap_experiment['half_max'] = np.log(0.5)/-bleach_recovery_params[2]
        frap_experiment['mob'] = -bleach_recovery_params[1] / (1-(bleach_recovery_params[0] + bleach_recovery_params[1] ))
    else: 
        frap_experiment['half_max'] = [np.array([np.log(0.5)/-bleach_recovery_params[3], np.log(0.5)/-(bleach_recovery_params[3]*bleach_recovery_params[4])])]
        frap_experiment['half_max_slow'] = np.log(0.5)/-bleach_recovery_params[3]
        frap_experiment['half_max_fast'] = np.log(0.5)/-(bleach_recovery_params[3]*bleach_recovery_params[4])

        frap_experiment['mob'] = -(bleach_recovery_params[1]+bleach_recovery_params[2]) / (1-(bleach_recovery_params[0] + bleach_recovery_params[1] + bleach_recovery_params[2]))
    
    #frap_experiment['mob'] = (bleach_recovery[bleach_recovery.size-1] - bleach_recovery[0])/(1 - bleach_recovery[0])

    frap_experiment['mob_corr'] = frap_experiment['mob'] / frap_experiment['gap_ratio']


    return roiData , frap_experiment    





