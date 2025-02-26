import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from skimage.morphology import disk
from skimage.filters import rank, threshold_otsu, gaussian





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

    #print(frap_experiment.wcell_corr.item())
    if frap_experiment.wcell_corr.item():
        wcellMask = find_wcell_roi(imageData[0:frap_experiment.bleach_frame.item(),:,:])
        wcellMean = np.zeros(imageData.shape[0])
        for t in range(imageData.shape[0]):
            wcellMean[t] = np.mean(imageData[t,wcellMask])  

        roiData['control_wcell'] = wcellMean
        frap_experiment['wcellMask'] = [wcellMask]

    for i,region in regionsInfo.iterrows():
        
        roi = imageData[:, int(regionsInfo.loc[i,'Y_roi']) : int(regionsInfo.loc[i,'Y_roi'] + regionsInfo.loc[i,'Height_roi']),
               int(regionsInfo.loc[i,'X_roi']) : int(regionsInfo.loc[i,'X_roi'] + regionsInfo.loc[i,'Width_roi'])]

        roiMean = np.mean(roi, axis = (1,2))

        if region['IsForBleach']:
            roiData.loc[0:roiMean.size, ['bleach']] = roiMean           

        else: 
            roiData.loc[0:roiMean.size, ['control_roi']] = roiMean

    if do_bkg:
        p5 = np.percentile(imageData, 5)
    

    return roiData, frap_experiment

# segment cell shape from a stack of frames
def find_wcell_roi(image):
    
    Tmean = np.mean(image, axis = 0, dtype = np.uint16)

    Tmean_f = rank.mean(Tmean, footprint = disk(3))
    Tmean_f = gaussian(Tmean, sigma=3)

    th = threshold_otsu(Tmean_f)
    mask =Tmean_f>th

    return mask
# Double exponential model for photobleaching decay
def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
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
    tau_fast = tau_slow*tau_multiplier
    return const + amp_slow*np.exp(-t/tau_slow) + amp_fast*np.exp(-t/tau_fast) 

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
    
    return const + amp*np.exp(-t/tau) 

# Calculate R-square value of the fit
def calculate_r_squared(data, fit_data):
    
    # residual sum of squares
    ss_res = np.sum((data - fit_data) ** 2)

    # total sum of squares
    ss_tot = np.sum((data - np.mean(data)) ** 2)

    # r-squared 
    r2 = 1 - (ss_res / ss_tot)

    return r2

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

    roiData['reference'] = roiData[ref_roi].to_numpy()
   
    # Use data from bleach timepoint+ delay to avoid including in the 
    # fitting timpoints showing a dip from recovery from photobleaching in the reference area
    bleach_data = roiData[ref_roi].iloc[frap_experiment.bleach_frame.item() + delay::].to_numpy()
    time_data = roiData['timestamp_frap'].iloc[frap_experiment.bleach_frame.item() + delay::].to_numpy()
  
    #Find fitting parameters for exponential function
    photobleach_decay_params = fit_photobleaching_exp(time_data, bleach_data, exp)

    #Calculate  decay curve for entire experiment
    photobleach_decay = estimate_exp_curve(roiData['timestamp_frap'], photobleach_decay_params, exp)

    #Calculate fitting error r_squared just for the timepoints used in the curve fitting step
    r_squared = calculate_r_squared(bleach_data,  estimate_exp_curve(time_data, photobleach_decay_params, exp))
    
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

    roiData['reference_decay_curve'] = photobleach_decay
    roiData['reference_synth'] = photobleach_decay
    roiData.loc[0:(frap_experiment.bleach_frame.item()-1),'reference_synth'] = roiData['reference'].iloc[0:(frap_experiment.bleach_frame.item())]
  
    roiData['bleach_photo_corr'] = roiData['bleach'] / (photobleach_decay/photobleach_decay[frap_experiment.bleach_frame.item() ])
    roiData['reference_photo_corr'] =  roiData['reference'] / (photobleach_decay/photobleach_decay[frap_experiment.bleach_frame.item() ])

    frap_experiment['photobleach_fit'] = [photobleach_decay_params]
    frap_experiment['photobleach_fit_r2'] = r_squared

    frap_experiment

    return roiData , frap_experiment    

#Peform Double normalization
def run_normalization(roiData, frap_experiment):
    frap_experiment['pre-reference'] = np.mean(roiData['reference_synth'].iloc[0:frap_experiment.bleach_frame.item()-1])
    frap_experiment['post-reference'] = np.mean(roiData['reference_synth'].iloc[frap_experiment.bleach_frame.item()])
    frap_experiment['pre-bleach'] = np.mean(roiData['bleach'].iloc[0:frap_experiment.bleach_frame.item()-1])
    frap_experiment['post_bleach'] = roiData['bleach'].iloc[frap_experiment.bleach_frame.item()]
    roiData['ref_norm'] = (frap_experiment['pre-reference'].iloc[0]/roiData['reference_synth'])
    #Double normalized curve (Prebleach set to 1)
    roiData['frap_norm'] = (frap_experiment['pre-reference'].iloc[0]/roiData['reference_synth']) \
         * (roiData['bleach']/frap_experiment['pre-bleach'].iloc[0])    

    #Full scale normalization
    roiData['frap_fullscale_norm'] = (roiData['frap_norm'] - roiData['frap_norm'].iloc[frap_experiment.bleach_frame.item()]) / \
    (np.mean(roiData['frap_norm'].iloc[0:frap_experiment.bleach_frame.item()-1])  - roiData['frap_norm'].iloc[frap_experiment.bleach_frame.item()] ) 
    
    frap_experiment['gap_ratio'] = frap_experiment['post-reference'] / frap_experiment['pre-reference']

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
    bleach_data = roiData['frap_norm'].iloc[frap_experiment.bleach_frame.item()::].to_numpy()
    time_data = roiData['timestamp_frap'].iloc[frap_experiment.bleach_frame.item()::].to_numpy()
  
   
    max_sig = np.max(bleach_data)

    if exp==1:
        inital_params = [max_sig/2, -max_sig/4, 1]
        bounds = ([0      , -max_sig,   0  ],
                 [ max_sig,        0,  1000])
        bleach_recovery_params, parm_cov = curve_fit(single_exponential, time_data, bleach_data, 
                                  p0=inital_params, bounds=bounds, maxfev=1000)
        bleach_recovery = single_exponential(time_data, *bleach_recovery_params)

        r_squared = calculate_r_squared(bleach_data,  single_exponential(time_data,*bleach_recovery_params))
    else:
        inital_params = [max_sig/2, -max_sig/4, -max_sig/4, 1, 0.1]
        bounds = ([0,     -max_sig,   -max_sig,     0,   0],
                [ max_sig,      0,            0,  100,   1])
        bleach_recovery_params, parm_cov = curve_fit(double_exponential, time_data, bleach_data, 
                                  p0=inital_params, bounds=bounds, maxfev=1000)
        bleach_recovery = double_exponential(time_data, *bleach_recovery_params)
        r_squared = calculate_r_squared(bleach_data,  double_exponential(time_data,*bleach_recovery_params))

    max_sig = np.max(bleach_data)

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

    roiData.loc[frap_experiment.bleach_frame.item()::,'bleach_recovery_curve'] = bleach_recovery
    roiData.loc[0:frap_experiment.bleach_frame.item()-1,'bleach_recovery_curve'] = 1

    frap_experiment['recovery_model'] = exp
    frap_experiment['recovery_fit'] = [bleach_recovery_params]
    frap_experiment['recovery_fit_r2'] = r_squared    
    frap_experiment['mob'] = -bleach_recovery_params[1] / (1-(bleach_recovery_params[0] + bleach_recovery_params[1] ))
    frap_experiment['mob'] = (bleach_recovery[bleach_recovery.size-1] - bleach_recovery[0])/(1 - bleach_recovery[0])

    frap_experiment['mob_corr'] = frap_experiment['mob'] / frap_experiment['gap_ratio']
    if exp==1:
        frap_experiment['half_max'] = np.log(0.5)/-bleach_recovery_params[2]
        #frap_experiment['mob'] = -bleach_recovery_params[1] / (1-(bleach_recovery_params[0] + bleach_recovery_params[1] ))
    else: 
        frap_experiment['half_max'] = [np.array([np.log(0.5)/-bleach_recovery_params[3], np.log(0.5)/-(bleach_recovery_params[3]*bleach_recovery_params[4])])]
        #frap_experiment['mob'] = -(bleach_recovery_params[1]+bleach_recovery_params[2]) / (1-(bleach_recovery_params[0] + bleach_recovery_params[1] + bleach_recovery_params[2]))
    
    return roiData , frap_experiment    





