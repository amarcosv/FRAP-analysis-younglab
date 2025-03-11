#from czitools.metadata_tools.czi_metadata import CziMetadata, get_metadata_as_object,obj2dict
from czitools.utils import misc
from pylibCZIrw import czi as pyczi
from aicsimageio import AICSImage
import xml.etree.ElementTree as ET
import pandas as pd
from io import StringIO
import numpy as np
import tools.processing_tools as processing_tools
import re
import os


def read_regions(cziPath):

    czi = AICSImage(cziPath)
    md_xml = ET.tostring(czi.metadata ,encoding="unicode")

    #with pyczi.open_czi(cziPath) as czidoc:
    #    md_xml = czidoc.raw_metadata
    #    root = ET.fromstring(md_xml)
    pxSize = czi.physical_pixel_sizes
    dimensions = czi.dims
    
    regions = pd.read_xml(StringIO(md_xml), xpath=".//RegionItem", attrs_only = True, parser="etree")
    annotations_id = pd.read_xml(StringIO(md_xml), xpath=".//Layers/Layer/Elements/Rectangle/Attributes", elems_only = False, parser="etree")   
    annotations = pd.read_xml(StringIO(md_xml), xpath=".//Layers/Layer/Elements/Rectangle/Geometry", elems_only = False, parser="etree")  
    bleach_timepoint = pd.read_xml(StringIO(md_xml), xpath=".//TimelineTrack[@Name='Bleaching Track']/TimelineElements/TimelineElement/Bounds", attrs_only = True, elems_only = False, parser="etree")    
    #print(bleach_timepoint['StartT'][0])
    for i,region in regions.iterrows():
        regions.loc[i,'bleach_frame'] = bleach_timepoint['StartT'].values
        if region['IsForBleach']:
            regions.loc[i, ['Color']] = '#FF0000'
        else:
            regions.loc[i, ['Color']] = '#32CD32'    

    for id in  enumerate(annotations_id['UniqueName']):
       #print(id[1])
       #print(regions['Key']==id[1])
       i=regions[regions['Key']==id[1]].index[0]
       #regions.insert(i, 'X_roi', annotations['Left'][i])
       regions.loc[i, 'X_roi'] = annotations['Left'][i]
       regions.loc[i, 'Y_roi'] = annotations['Top'][i]
       regions.loc[i, 'Width_roi'] = annotations['Width'][i]
       regions.loc[i, 'Height_roi'] = annotations['Height'][i]

    
    regions['Height_px'] = regions['Height'] / pxSize.Y 
    regions['Width_px'] = regions['Width'] / pxSize.X 
    regions['X_px'] = (regions['X']  / pxSize.X) + dimensions.X/2 - (regions['Width_px']/2)
    regions['Y_px'] = (regions['Y']  / pxSize.Y) + dimensions.Y/2 + (regions['Height_px']/2)
    #regions['bleach_frame'] = bleach_timepoint
    regions.drop(['GraphicProperties','ItemIndex', 'IsSelected', 'IsProtected', 'IsForAnalysis', 'IsForAcquisition'], axis=1, inplace=True)
    return regions

def load_image_data(cziPath):

    with pyczi.open_czi(cziPath) as czidoc:
        # get the image dimensions as a dictionary, where the key identifies the dimension
        total_bounding_box = czidoc.total_bounding_box
        print('[load_image_data] Loading dataset with dimensions: ' + str(czidoc.total_bounding_box))
        #print(czidoc.total_bounding_box)

        frame_limit = total_bounding_box["T"][1]
        #frame_limit = int(frame_limit/2)
              
        img_data = np.zeros((frame_limit,total_bounding_box["Y"][1]  ,total_bounding_box["X"][1]))

        

        for t in range(frame_limit):
            
            # read a 2D image plane and optionally specify planes, zoom levels and ROIs
            img_data[t,:,:] = np.squeeze(czidoc.read(plane={"T": t}, zoom=1.0))

    return img_data

def load_frame_metadata(cziPath):
    #Command from https://github.com/sebi06/czitools/blob/abae690eb1b3c4ac31054e578243352ee0b106f5/demo/notebooks/read_czi_metadata.ipynb
    # get the planetable for the CZI file
    pt = misc.get_planetable(cziPath,
                         norm_time=True,
                         pt_complete=True,
                         t=0,
                         c=0,
                         z=0)
    pt.drop(['Subblock', 'T', 'Z', 'C', 'xstart', 'ystart', 'width', 'height', 'Scene'], axis=1, inplace=True)
    return pt



def parse_filename(filename):
    #DISHXX_PROT_CONDITION_ROI read from folder

    prefix = filename.split('Airyscan')

    if "wt" in prefix[0].lower():
        group = "WT"
    elif "mut" in prefix[0].lower():
        group = "MUT"
    else:
        group  = "unknown"

    params = re.split('[_-]', prefix[0])
    #print(params)
    if len(params)>=3:
        dishN = params[0].replace("dish", "")
        prot = (params[1].replace("WT", "")).replace("MUT", "")
        if "roi" in params[2]:
            roi = params[2]
        elif "roi" in params[3]:
            roi = params[3]
        else:
            roi = "null"
    else:
        dishN = ""
        prot = "unknown"
        roi = ""

    print ("[parse_filename] Metadata retreived from filename: ")
    print ("\tgroup = " + group)
    print("\tdish = " + dishN )
    print("\tProtein = " + prot )
    print("\tROI  = " + roi )

    return group, dishN, prot, roi

def saveResults(datasetPath, roiData, frap_experiment):
    OUTPUT_FOLDER, foldername = os.path.split(datasetPath)
    basename = frap_experiment['protein'].iloc[0] + '_' + frap_experiment['group'].iloc[0] 
    #print(fname)

    print("[saveResults] Saving results to file")
    print("\tTimepoint data: " +  os.path.join(OUTPUT_FOLDER, basename +'_roiData'+ '.csv'))
    print("\tFRAP analysis summary: " +  os.path.join(OUTPUT_FOLDER, basename +'_frap_summary'+ '.csv'))

    roiData.to_csv( os.path.join(OUTPUT_FOLDER, basename +'_roiData'+ '.csv'))    
    frap_experiment.to_csv(  os.path.join(OUTPUT_FOLDER, basename +'_frap_summary'+ '.csv'))