import io_tools
import processing_tools
import pandas as pd
import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def import_FRAP_data(cziPath, wcell_corr= False):
    regions = io_tools.read_regions(cziPath)
    imageData = io_tools.load_image_data(cziPath)
    frames_metadata = io_tools.load_frame_metadata(cziPath)
    #time_diffs = np.diff(time_vector)

    frap_experiment = pd.DataFrame([int(regions.loc[0,'bleach_frame'].item())], columns = ['bleach_frame']) 
    frap_experiment['wcell_corr'] = wcell_corr       
    #frap_experiment['bleach_frame'] = int(regions.loc[0,'bleach_frame'].item())
           
    roiData, frap_experiment = processing_tools.process_ROI(imageData,frap_experiment, regions, frames_metadata,False)




    return roiData,regions, imageData, frap_experiment

def run_FRAP_analysis(roiData, frap_experiment, fitting_exp = 1):

    if frap_experiment.wcell_corr.item():
        ref_roi = 'control_wcell'
    else:
        ref_roi = 'control_roi'

    roiData, frap_experiment = processing_tools.photobleaching_corr(roiData, ref_roi , frap_experiment, delay = 10,exp=1)

    #roiData, frap_experiment = processing_tools.pre_bleach_normalization(roiData , frap_experiment)

    roiData, frap_experiment = processing_tools.run_normalization(roiData , frap_experiment)

    roiData, frap_experiment = processing_tools.fit_recovery_curve(roiData, frap_experiment, fitting_exp)

    return roiData, frap_experiment

#def perform_photobleaching_correction(roiData, frap_experiment, delay = 3, exp=1, use_wcell = True):


def process_FRAP_folder(folderPath, wcell_corr= True, fitting_exp = 1):
    print('Processing files from directory: ' + folderPath)

    fileList = [f for f in os.listdir(os.path.join(folderPath))  if f.endswith('.czi')]
    basenames = [f.split('.czi')[0] for f in os.listdir(os.path.join(folderPath))  if f.endswith('.czi')]  

    #print(fileList)

    dataset_roiData = []
    dataset_frap_experiment = []
    
    
 
    #plt.ioff()
    fig, ax_previews = plt.subplots(nrows = int(np.ceil(len(fileList)/2)),ncols = 2, figsize=(12, 6*np.ceil(len(fileList)/2)))


    for idx,f in enumerate(fileList):
        print('\n[processFiles] processing file ' + str(idx+1) + ' of ' +str(len(fileList)))
        print('[processFiles] Filename = ' + f)

        group, dish, prot, roi = io_tools.parse_filename(f)
 


        roiData,regions, image, frap_experiment = import_FRAP_data(os.path.join(folderPath,f), wcell_corr= wcell_corr)
        roiData, frap_experiment = run_FRAP_analysis(roiData, frap_experiment, fitting_exp)

        #if idx==0:
            #imageData = np.zeros([len(fileList),image.shape[0], image.shape[1]])

        #imageData[idx,:,:] = image
        roiData.insert(loc=0, column = 'file', value = basenames[idx])
        roiData.insert(loc=1, column = 'group', value = group)
        roiData.insert(loc=2, column = 'dish', value = dish)
        roiData.insert(loc=3, column = 'protein', value = prot)
        roiData.insert(loc=4, column = 'roiN', value = roi)        
        frap_experiment.insert(loc=0, column = 'file', value = basenames[idx])
        frap_experiment.insert(loc=1, column = 'group', value = group)
        frap_experiment.insert(loc=2, column = 'dish', value = dish)
        frap_experiment.insert(loc=3, column = 'protein', value = prot)
        frap_experiment.insert(loc=4, column = 'roiN', value = roi)
        #print(str(idx // 2) + "  " + str(idx % 2) )
        if len(fileList) > 2:
            ax = ax_previews[idx // 2, idx % 2]
        else:
            ax = ax_previews[idx]
        if frap_experiment.wcell_corr.item():
             generate_preview(ax, image, regions, frap_experiment['wcellMask'].values[0])
        else: 
             generate_preview(ax,image, regions)
        ax.set_title(basenames[idx])
        
        dataset_roiData.append(roiData)
        dataset_frap_experiment.append(frap_experiment)
    
    dataset_frap_experiment = pd.concat(dataset_frap_experiment, ignore_index=True)

    bleach_frame = dataset_frap_experiment['bleach_frame'][0]       
    nframes = dataset_frap_experiment['nframes'][0]
    dt= np.mean(dataset_frap_experiment['dt'])

    dataset_roiData = rebin_results(dataset_roiData, dt ,bleach_frame , nframes )

    dataset_roiData = pd.concat(dataset_roiData, ignore_index=True)

    #plt.close(fig)

    return dataset_roiData, dataset_frap_experiment, fig

    
def rebin_results(dataset_roiData, dt, frap_frame, n):

    end = 0 + (n - 1) * dt
    timestamps = np.linspace(0, end, n) 
    timestamps = timestamps-timestamps[frap_frame]
    print(len(timestamps))
    
    rebinned_dataset_roiData = []

    for idx,roiData in enumerate(dataset_roiData):
        rebinned_roiData = pd.DataFrame(columns = ['timestamp','timestamp_frap', 'frap', 'control', 'bkg'])
        dataset_roiData[idx]['timestamp_frap_r'] = timestamps
        #dataset_roiData[idx]['frap_norm_r']
        #b = roiData['frap_norm'].to_numpy(dtype=np.float32)
        dataset_roiData[idx]['frap_norm_r'] = np.interp(timestamps, roiData['timestamp_frap'].to_numpy(dtype=np.float64), roiData['frap_norm'].to_numpy(dtype=np.float64))
        dataset_roiData[idx]['frap_fullscale_norm_r'] = np.interp(timestamps, roiData['timestamp_frap'].to_numpy(dtype=np.float64), roiData['frap_fullscale_norm'].to_numpy(dtype=np.float64))


        print(roiData['file'][0])
  
    return dataset_roiData

def generate_preview(ax, image, regions, wcell_mask = None):
    
    rect = patches.Rectangle((regions.X_roi[1], regions.Y_roi[1]), regions.Width_roi[1], regions.Height_roi[1], linewidth=2, 
                         edgecolor=regions.Color[1], facecolor="none")
 
    rect2 = patches.Rectangle((regions.X_roi[0], regions.Y_roi[0]), regions.Width_roi[0], regions.Height_roi[0], linewidth=2, 
                         edgecolor=regions.Color[0], facecolor="none")
    
    image = np.mean(image[0:int(regions.loc[0,'bleach_frame']),:,:], axis = 0, dtype = np.uint16)

    #fig, ax = plt.subplots()
    ax.imshow(image)

    if wcell_mask is not None:        
        ax.imshow(wcell_mask.astype(int), cmap='Greens', alpha=0.2) 

    ax.add_patch(rect)
    ax.add_patch(rect2)    
    #plt.close(fig)
    #return ax