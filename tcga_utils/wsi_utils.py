# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Functions for dealing with whole slide images - Based on code from https://github.com/deroneriksson/python-wsi-preprocessing """

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import pandas as pd
import numpy as np
import requests
import json
import re
import openslide
import PIL
import math
import cv2

from skimage.morphology import remove_small_objects
from openslide.deepzoom import DeepZoomGenerator

def open_slide(svs_filename):
    """
    Open a wsi and return a OpenSlide object
    """
    return openslide.OpenSlide(svs_filename)

def slide_to_img(slide, new_mpp=0.5, return_np=True, return_sizes=False):
    """
    Scale slide image based on desired microns per pixel
    """
    old_mpp_x = np.float(slide.properties['openslide.mpp-x'])
    old_mpp_y = np.float(slide.properties['openslide.mpp-y'])
    
    new_mpp = np.float(new_mpp)
    
    SCALE_FACTOR = new_mpp/old_mpp_x
        
    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)

    if return_np:
        img = np.array(img)

    if return_sizes:
        return img, large_w, large_h, new_w, new_h

    else: 
        return img

def slide_to_tiles(slide, new_mpp=0.5, tile_size=512, overlap=0):
    """
    Convert a slide to tiles with a given mpp and tile size
    """

    # Identify appropriate pyramid level for given micron per pixel
    mpp = np.float(slide.properties['openslide.mpp-x'])
    scale_factor = new_mpp/mpp
    offset = math.floor(np.log2((scale_factor + 0.1)))
    

    level_mpp = 2**offset * mpp
    scale_from_level = new_mpp/level_mpp
    level_tile_size = math.ceil(tile_size * scale_from_level)

    generator = DeepZoomGenerator(slide, 
                                tile_size=level_tile_size, 
                                overlap=overlap, 
                                limit_bounds=True)

    highest_level = generator.level_count - 1 
    level = highest_level - offset

    cols, rows = generator.level_tiles[level]
    
    # Extract tiles
    tiles = []
    for col in range(cols):
        for row in range(rows):
            tile = np.array(generator.get_tile(level, (col, row)))
            if tile.shape[0] == level_tile_size and tile.shape[1] == level_tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size), interpolation = cv2.INTER_LINEAR)
                tiles.append(tile)
    tiles = np.stack(tiles, axis=0)

    return tiles

def rgb_to_grayscale(rgb_img, invert=True, dtype='uint8'):
    """
    Convert rgb numpy array to grey scale
    """

    if dtype not in ['uint8', 'float', 'float32']:
        raise ValueError('dtype should be uint8, float, or float32')

    color_weights = [0.2125, 0.7154, 0.114]
    grayscale = np.dot(rgb_img, color_weights)
    grayscale = grayscale.astype(dtype)

    if invert:
        if dtype == 'uint8':
            grayscale = 255 - grayscale
        elif dtype == 'float' or dtype == 'float32':
            grayscale = 1 - grayscale

    return grayscale

def threshold(img, dtype='uint8', mode='hysteresis', **kwargs):
    """
    Threshold image
    """
    if dtype not in ['uint8', 'float', 'float32', 'bool']:
        raise ValueError('dtype should be bool, uint8, float, or float32')

    if mode not in ['hysteresis', 'otsu', 'local_otsu']:
        raise ValueError('mode should be hysteresis, otsu, or local_otsu')

    if mode == 'hysteresis':
        low = kwargs.get('low', 50)
        high = kwargs.get('high',100)
        mask = skimage.filters.apply_hysteresis_threshold(img, low, high)
    elif mode == 'otsu':
        threshold = skimage.filters.threshold_otsu(img)
        mask = (img > threshold)
    elif mode == 'local_otsu':
        disk_size = kwargs.get('disk_size', 3)
        mask = skimage.filters.rank.otsu(img, skimage.morphology.disk(disk_size))

    if dtype == 'bool':
        pass
    elif dtype == 'float' or dtype == 'float32':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = mask.astype(dtype) * 255

    return mask

def adjust_contrast(img, mode='stretch', dtype='uint8', **kwargs):
    """
    Adjust contrast
    """
    if dtype not in ['uint8', 'float', 'float32', 'bool']:
        raise ValueError('dtype should be bool, uint8, float, or float32')

    if mode not in ['stretch', 'hist', 'adapthist']:
        raise ValueError('mode should be stretch, hist, or adapthist')

    if mode == 'stretch':
        low = kwargs.get('low', 40)
        high = kwargs.get('high', 60)

        low_p, high_p = np.percentile(img, low * 100.0/255.0, high *100.0/255.0)
        img = skimage.exposure.rescale_intensity(img, in_range=(low_p, high_p))
    elif mode == 'hist':
        nbins = kwargs.get('nbins', 256)
        if img.dtype == 'uint8' and nbins !=256:    # If uint8 and nbins is specified, convert to float
            img = img.astype('float')/255
        img = skimage.exposure.equalize_hist(img, nbins=nbins)
    elif mode == 'adapt_hist':
        nbins = kwargs.get('nbins', 256)
        clip_limit = kwargs.get('clip_limit', 0.01)
        img = skimage.exposure.equalize_adapthist(img, nbins=nbins, clip_limit=clip_limit)

    if dtype == 'uint8':
        img = (img*255).astype(dtype)
    else:
        img = img.astype(dtype)

    return img

def filter_red(img, red_lower_thresh, 
                    green_upper_thresh, 
                    blue_upper_thresh,
                    dtype='bool'):
    """
    Create a mask that filters out red pixels
    """
    r = img[...,0] > red_lower_thresh
    g = img[...,1] < green_upper_thresh
    b = img[...,2] < blue_upper_thresh

    mask = ~(r & g & b)

    if dtype == 'bool':
        pass
    elif dtype == 'float':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = (255*mask).astype(dtype)

    return mask

def filter_green(img, red_upper_thresh, 
                    green_lower_thresh,
                    blue_lower_thresh,
                    dtype='bool'):
    """
    Create a mask that filters out green pixels
    """
    r = img[...,0] < red_upper_thresh
    g = img[...,1] > green_lower_thresh
    b = img[...,2] > blue_lower_thresh

    mask = ~(r & g & b)

    if dtype == 'bool':
        pass
    elif dtype == 'float':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = (255*mask).astype(dtype)

    return mask

def filter_blue(img, red_upper_thresh,
                    green_upper_thresh,
                    blue_lower_thresh,
                    dtype='bool'):
    """
    Create a mask that filters out blue pixels
    """
    r = img[...,0] < red_upper_thresh
    g = img[...,1] < green_upper_thresh
    b = img[...,2] > blue_lower_thresh

    mask = ~(r & g & b)

    if dtype == 'bool':
        pass
    elif dtype == 'float':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = (255*mask).astype(dtype)

    return mask

def filter_grays(img, tolerance=15, dtype="bool"):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.
    Args:
        img: RGB image as a NumPy array.
        tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
        dtype: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    (h, w, c) = img.shape

    img = img.astype(np.int)
    rg_diff = abs(img[...,0] - img[...,1]) <= tolerance
    rb_diff = abs(img[...,0] - img[...,2]) <= tolerance
    gb_diff = abs(img[...,1] - img[...,2]) <= tolerance
    mask = ~(rg_diff & rb_diff & gb_diff)

    if dtype == 'bool':
        pass
    elif dtype == 'float':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = (255*mask).astype(dtype)

    return mask

def filter_green_channel(img, green_thresh=200, 
                            avoid_overmask=True, 
                            overmask_thresh=90, 
                            dtype="bool"):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
    and eosin are purplish and pinkish, which do not have much green to them.
    Args:
        np_img: RGB image as a NumPy array.
        green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
        avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
        overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
    """

    g = img[:, :, 1]
    mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        print(
          "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
            mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
        mask = filter_green_channel(img, new_green_thresh, avoid_overmask, overmask_thresh, dtype)

    if dtype == 'bool':
        pass
    elif dtype == 'float':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = (255*mask).astype(dtype)

    return mask

def filter_red_pen(img, dtype='bool'):
    """
    Create a mask to filter out red pen marks from a slide.
    Args:
        rgb: RGB image as a NumPy array.
        dtype: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing the mask.
    """
    mask = filter_red(img, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(img, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(img, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(img, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(img, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(img, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(img, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
           filter_red(img, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(img, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    
    if dtype == 'bool':
        pass
    elif dtype == 'float':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = (255*mask).astype(dtype)

    return mask

def filter_green_pen(img, dtype="bool"):
    """
    Create a mask to filter out green pen marks from a slide.
    Args:
        rgb: RGB image as a NumPy array.
        dtype: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing the mask.
    """
    mask = filter_green(img, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(img, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(img, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(img, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(img, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(img, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(img, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(img, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(img, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(img, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(img, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(img, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(img, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(img, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(img, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)

    if dtype == 'bool':
        pass
    elif dtype == 'float':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = (255*mask).astype(dtype)

    return mask

def filter_blue_pen(img, dtype="bool"):
    """
    Create a mask to filter out blue pen marks from a slide.
    Args:
        rgb: RGB image as a NumPy array.
        dtype: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing the mask.
    """
    mask = filter_blue(img, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(img, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(img, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(img, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(img, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(img, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(img, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(img, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(img, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(img, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(img, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(img, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)

    if dtype == 'bool':
        pass
    elif dtype == 'float':
        mask = mask.astype(dtype)
    elif dtype == 'uint8':
        mask = (255*mask).astype(dtype)

    return mask

def rgb_to_hed(img, dtype='uint8'):
    img = skimage.color.rgb2hed(img)
    if dtype == 'float' or dtype == 'float32':
        img = skimage.exposure.rescale_intensity(img, out_range=(0.0, 1.0)).astype(dtype)
    elif dtype == 'uint8':
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255)).astype(dtype)
    return img

def rgb_to_hsv(img, dtype='uint8'):
    img = skimage.color.rgb2hsv(img)
    return img

def mask_percent(img):
    """
    Compute the percentage of an image that is masked (has mask value 0)
    """
    if (len(img.shape) == 3) and (img.shape[2] == 3):
        img = np.sum(img, axis=-1)

    mask_percentage = 100 * (1 - np.count_nonzero(img) / img.size )
    return mask_percentage

def tissue_percent(img):
    """
    Compute the percentage of an image that is tissue (i.e. not masked)
    """
    return 100 - mask_percent(img)

def filter_tile(img):
    """
    Compute a tissue mask for a given tile using color filters
    """
    mask_not_green = filter_green_channel(img, avoid_overmask = False)
    mask_not_gray = filter_grays(img)
    mask_no_red_pen = filter_red_pen(img)
    mask_no_green_pen = filter_green_pen(img)
    mask_no_blue_pen = filter_blue_pen(img)
    
    mask = mask_not_green & mask_not_gray & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
    
    # Remove small objects
    mask = remove_small_objects(mask, min_size=3000)
    
    return mask

def filter_tiles(tiles, tissue_threshold=50, mask_tissue=False):
    """
    Apply filter_tile to a collection of tiles and return only the
    tiles whose tissue percentage exceeds the given threshold
    """
    filtered_tiles = []
    for tile in tiles:
        mask = filter_tile(tile)
        tissue_percentage = tissue_percent(mask)
        if mask_tissue:
            tile = tile * mask
        if tissue_percentage > tissue_threshold:
            filtered_tiles.append(tile)
    filtered_tiles = np.stack(filtered_tiles, axis=0)
    
    if mask_tissue:
        return filtered_tiles, mask, mask_tissue
    else:
        return filtered_tiles    

"""
Prototypes - untested and should not be used
"""

class TCGADataset(object):
    """
    Prototype TCGA Dataset object to handle file download conversion to 
    ML friendly tile format
    """
    def __init__(self, uuid, mpp=0.5):
        svs_filename = download_utils.download_by_uuids(uuid)
        slide = wsi_utils.open_slide(svs_filename)
        tiles = wsi_utils.slide_to_tiles(slide, new_mpp=mpp)
        filtered_tiles = wsi_utils.filter_tiles(tiles)
        download_utils.remove_file(svs_filename)
        
        self.uuid = uuid
        self.tiles = filtered_tiles
        self.mpp = mpp
        self.annotation = {}
        return None

    def _add_annotation(annotation_dict, annotation_name):
        """
        Add annotation from an annotation dictionary. Assumues the uuid is the key 
        for the annotation
        """
        self.annotation[annotation_name] = annotation_dict[self.uuid]
