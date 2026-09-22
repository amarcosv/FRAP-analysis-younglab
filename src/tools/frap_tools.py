import tools.io_tools as io_tools
import tools.processing_tools as processing_tools
import pandas as pd
import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

REF_DELAY = 10


def import_FRAP_data(cziPath, wcell_corr= False):

    root, extension = os.path.splitext(cziPath)

    if extension == '.czi':
        regions = io_tools.read_regions(cziPath)
        imageData = io_tools.load_image_data(cziPath)
        frames_metadata = io_tools.load_frame_metadata(cziPath)
        frap_experiment = pd.DataFrame([int(regions.loc[0,'bleach_frame'].item())], columns = ['bleach_frame']) 
    
        frap_experiment['wcell_corr'] = wcell_corr 
            
        roiData, frap_experiment = processing_tools.process_ROI(imageData,frap_experiment, regions, frames_metadata,False)

        return roiData, frap_experiment, regions, imageData

    elif extension == '.csv':

        roiData, bleach_frame = io_tools.read_zeiss_CSV(cziPath)
        frap_experiment = pd.DataFrame({'bleach_frame':[bleach_frame],'wcell_corr' : [False]})
        frap_experiment['dt'] = np.median(np.diff(roiData['timestamp'] [frap_experiment.bleach_frame.item()::]))
        frap_experiment['nframes'] = len(roiData['timestamp'])  
        imageData = []
        regions = []

        return roiData, frap_experiment, [],[]

    

def run_FRAP_analysis(roiData, frap_experiment, fitting_exp = 1):

    if frap_experiment.wcell_corr.item():
        ref_roi = 'control_wcell'
    else:
        ref_roi = 'control_roi'

    roiData, frap_experiment = processing_tools.photobleaching_corr(roiData, ref_roi , frap_experiment, delay = REF_DELAY,exp=1)

    #roiData, frap_experiment = processing_tools.pre_bleach_normalization(roiData , frap_experiment)

    roiData, frap_experiment = processing_tools.run_double_normalization(roiData , frap_experiment)

    roiData, frap_experiment = processing_tools.fit_recovery_curve(roiData, frap_experiment, fitting_exp)

    return roiData, frap_experiment

#def perform_photobleaching_correction(roiData, frap_experiment, delay = 3, exp=1, use_wcell = True):


def process_FRAP_folder(folderPath, wcell_corr= True, fitting_exp = 1, output_path=None):
    print('Processing files from directory: ' + folderPath)

    fileList = [f for f in os.listdir(os.path.join(folderPath))  if f.endswith(('.czi','.csv'))]
    basenames = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(folderPath))  if f.endswith(('.czi','.csv'))]

    #print(fileList)
    do_preview = True
    if '.csv' in fileList[0]:
        do_preview = False

    dataset_roiData = []
    dataset_frap_experiment = []
    failures = []



    #plt.ioff()
    fig = []
    if do_preview:
        fig, ax_previews = plt.subplots(nrows = int(np.ceil(len(fileList)/2)),ncols = 2, figsize=(12, 6*np.ceil(len(fileList)/2)))


    for idx,f in enumerate(fileList):
        print('\n[processFiles] processing file ' + str(idx+1) + ' of ' +str(len(fileList)))
        print('[processFiles] Filename = ' + f)

        try:
            group, dish, roi, dose = io_tools.parse_filename(f)



            roiData,frap_experiment, regions, image,  = import_FRAP_data(os.path.join(folderPath,f), wcell_corr= wcell_corr)
            roiData, frap_experiment = run_FRAP_analysis(roiData, frap_experiment, fitting_exp)

            #if idx==0:
                #imageData = np.zeros([len(fileList),image.shape[0], image.shape[1]])

            #imageData[idx,:,:] = image
            roiData.insert(loc=0, column = 'file', value = basenames[idx])
            roiData.insert(loc=1, column = 'group', value = group)
            roiData.insert(loc=2, column = 'dish', value = dish)
            roiData.insert(loc=3, column = 'dose', value = dose)
            roiData.insert(loc=4, column = 'roiN', value = roi)
            frap_experiment.insert(loc=0, column = 'file', value = basenames[idx])
            frap_experiment.insert(loc=1, column = 'group', value = group)
            frap_experiment.insert(loc=2, column = 'dish', value = dish)
            frap_experiment.insert(loc=3, column = 'dose', value = dose)
            frap_experiment.insert(loc=4, column = 'roiN', value = roi)

            if do_preview:
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
        except Exception as e:
            print('WARNING: [processFiles] Failed to process file ' + f + ': ' + str(e))
            failures.append({'file': basenames[idx], 'error': str(e)})
            continue

    if not dataset_roiData:
        raise RuntimeError('[process_FRAP_folder] All ' + str(len(fileList)) + ' file(s) in ' + folderPath + ' failed to process')

    if output_path is not None:
        io_tools.save_failed_files(os.path.join(output_path, os.path.basename(folderPath)), failures)
    elif failures:
        print('[process_FRAP_folder] ' + str(len(failures)) + ' file(s) failed - see warnings above')

    dataset_frap_experiment = pd.concat(dataset_frap_experiment, ignore_index=True)
    #print(dataset_frap_experiment)
    dt = dataset_frap_experiment['dt'].median()

    dataset_roiData = pd.concat(dataset_roiData, ignore_index=True)

    # Resample each file's recovery curve onto a shared dt-multiple grid (real interpolation)
    dataset_roiData = rebin_results(dataset_roiData, dt)

    #plt.close(fig)

    return dataset_roiData, dataset_frap_experiment, fig

# Resample each file's recovery curve onto a shared dt-multiple grid.
# Each real timestamp is snapped to its nearest common-dt slot and the value
# is interpolated there from that file's own real data only, never beyond
# that file's own recorded range.
def rebin_results(dataset_roiData, dt):

    for file_id, group in dataset_roiData.groupby('file', sort=False):
        group = group.sort_values('timestamp_frap')
        t_raw = group['timestamp_frap'].to_numpy(dtype=np.float64)

        snapped_t = np.round(t_raw / dt) * dt

        frap_norm_r = np.interp(snapped_t, t_raw, group['frap_norm'].to_numpy(dtype=np.float64))
        frap_fullscale_norm_r = np.interp(snapped_t, t_raw, group['frap_fullscale_norm'].to_numpy(dtype=np.float64))

        dataset_roiData.loc[group.index, 'timestamp_frap_r'] = snapped_t
        dataset_roiData.loc[group.index, 'frap_norm_r'] = frap_norm_r
        dataset_roiData.loc[group.index, 'frap_fullscale_norm_r'] = frap_fullscale_norm_r

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