#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:29:48 2026

@author: stavis
"""

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import Image3DModel, PointsModel
from spatialdata.transformations import (Identity, 
                                         Translation, 
                                         set_transformation, 
                                         get_transformation,
                                         Sequence)
from skimage.io import imread
from spatialdata.transformations import align_elements_using_landmarks

def register(ref_pts, mov_pts, mov_elm):
    #get target registration points
    kinsight_pts_df = pd.read_csv(ref_pts)
    kinsight_pts = PointsModel().parse(np.array(list(zip(kinsight_pts_df['axis-3'],
                                                         kinsight_pts_df['axis-2']))),
                                       transformations = {'kinsight_2D':Identity()})
    data[ref_pts] = kinsight_pts
    data.write_element(ref_pts)
    
    #get subject registration points
    new_pts_df = pd.read_csv(mov_pts)
    new_pts = PointsModel().parse(np.array(list(zip(new_pts_df['axis-1'], 
                                                    new_pts_df['axis-0']))),
                                       transformations = {'unaligned_2D':Identity()})
    data[mov_pts] = new_pts
    data.write_element(mov_pts)
    
    #calculate a 2D alignment
    affine = align_elements_using_landmarks(references_coords=data[ref_pts],
                                            moving_coords=data[mov_pts],
                                            reference_element=data[ref_pts],
                                            moving_element=data[mov_pts],
                                            reference_coordinate_system='kinsight_2D',
                                            moving_coordinate_system='unaligned_2D',
                                            new_coordinate_system='aligned_2D')
    #add the 3rd dimension shift to the alignment
    zshift = get_transformation(data[mov_elm], 'unaligned')
    tform = Sequence([affine, zshift])
    set_transformation(data[mov_elm],
                       tform,
                       to_coordinate_system = 'aligned')
    data.write_transformations()


# =============================================================================
# kinsight
# =============================================================================
tif = 'z-stack_kinsight.tif'
kinsight = imread(tif)[np.newaxis, :, :, :]
kinsight = Image3DModel().parse(kinsight,
                                transformations={'kinsight':Identity(),
                                                 'aligned':Identity()})
data = SpatialData(images = {'kinsight':kinsight})
data.write('combined_data.zarr', overwrite = True)

# =============================================================================
# MS data
# =============================================================================
#The MS data are treated as 2D images with a high channel count
#we need to encode the analyte information as strings in order to interpret these channels
features = pd.read_csv('combined_slices_features.csv', comment = '#', sep = ';')
features = features[np.isfinite(features['1/K0 [V·s/cm²]'])]
analytes = [f'm/z:{mz}, 1/K0:{im}, CCS:{ccs}' for mz,im,ccs in zip(features['m/z'],
                                                                   features['1/K0 [V·s/cm²]'],
                                                                   features['CCS [Å²]'])]

ms_tifs = (('205.ome.tif',242),
           ('slide_89.ome.tif',126),
           ('slide_1_2.ome.tif',288),
           ('161.ome.tif',198),
           ('129.ome.tif',165))
for tif, z in ms_tifs:
    #parse raw data
    img = imread(tif)[:,np.newaxis,:,:]
    img = Image3DModel().parse(img,
                               transformations={'unaligned':Translation([z], 'z')},
                               c_coords = analytes)
    data[tif] = img
    data.write_element(tif)

    #registration
    register(f'{tif}_kinsight_pts.csv', 
             f'{tif}_MS_pts.csv',
             tif)

# =============================================================================
# unstained brightfield
# =============================================================================
white_tifs = (('position_128_white_check.tif', 126),
              ('position_294_white_check.tif', 288),
              ('position_168_white_check.tif', 165),
              ('position_247_white_check.tif', 242),
              ('position_202_white_check.tif', 198))
for tif, z in white_tifs:
    #parse raw data
    img = imread(tif)[:,np.newaxis,:,:]
    img = np.swapaxes(img, 0, -1)
    img = np.swapaxes(img, -2, -1)
    img = Image3DModel().parse(img,
                               transformations={'unaligned':Translation([z], 'z')})
    data[tif] = img
    data.write_element(tif)

    #registration
    register(f'{tif}_kinsight_pts.csv', 
             f'{tif}_white_pts.csv',
             tif)

# =============================================================================
# H&E
# =============================================================================
he_tifs = (('position_128_HE_check.tif', 126),
           ('position_294_HE_check.tif', 288),
           ('position_168_HE_check.tif', 165),
           ('position_247_HE_check.tif', 242))
for tif, z in he_tifs:
    #parse raw data
    img = imread(tif)[:,np.newaxis,:,:]
    img = np.swapaxes(img, 0, -1)
    img = np.swapaxes(img, -2, -1)
    img = Image3DModel().parse(img,
                               transformations={'unaligned':Translation([z], 'z')})
    data[tif] = img
    data.write_element(tif)

    #registration
    register(f'{tif}_kinsight_pts.csv', 
             f'{tif}_HE_pts.csv',
             tif)

## register H&E position 202 by aligning to position 202 pre-stained image, then
## transferring its transformation to the H&E image
#parse raw data
tif = 'position_202_HE_check.tif'
z = 198
img = imread(tif)[:,np.newaxis,:,:]
img = np.swapaxes(img, 0, -1)
img = np.swapaxes(img, -2, -1)
img = Image3DModel().parse(img,
                           transformations={'unaligned':Translation([z], 'z')})
data[tif] = img
data.write_element(tif)

#get target registraiton points
ref_pts = 'position_202_white_check.tif_ptsForHE.csv'
white_pts_df = pd.read_csv(ref_pts)
white_pts = PointsModel().parse(np.array(list(zip(white_pts_df['axis-1'],
                                                  white_pts_df['axis-0']))),
                                   transformations = {'kinsight_2D':Identity()})
data[ref_pts] = white_pts
data.write_element(ref_pts)

#get subject registration points
mov_pts = 'position_202_HE_check.tif_ptsForHE.csv'
HE_pts_df = pd.read_csv(mov_pts)
HE_pts = PointsModel().parse(np.array(list(zip(HE_pts_df['axis-1'], 
                                               HE_pts_df['axis-0']))),
                                   transformations = {'unaligned_2D':Identity()})
data[mov_pts] = HE_pts
data.write_element(mov_pts)

#calculate a 2D alignment to the white image
affine = align_elements_using_landmarks(references_coords=data[ref_pts],
                                        moving_coords=data[mov_pts],
                                        reference_element=data[ref_pts],
                                        moving_element=data[mov_pts],
                                        reference_coordinate_system='kinsight_2D',
                                        moving_coordinate_system='unaligned_2D',
                                        new_coordinate_system='aligned_2D')
#add the 3D alignment of the white image to the total transform
white_tform = get_transformation(data['position_202_white_check.tif'], 'aligned')
tform = Sequence([affine, white_tform])
set_transformation(data[tif],
                   tform,
                   to_coordinate_system = 'aligned')
data.write_transformations()





