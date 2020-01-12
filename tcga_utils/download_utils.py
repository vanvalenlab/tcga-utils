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
""" Functions for dealing with tcga data - Based on code from https://github.com/deroneriksson/python-wsi-preprocessing """

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import pandas as pd
import requests
import json
import re

def load_manifest_file(manifest_file):
    """ 
    Parse manifest file, return pandas dataframe
    """
    df = pd.read_csv(manifest_file, sep='\t')
    return df

def download_by_uuids(uuids, save_direc='/data/TCGA'):
    """
    Download data given uuids, return filename
    """
    if isinstance(uuids, str):
        uuids = [uuids]
    elif isinstance(uuids, list):
        pass
    else:
        raise ValueError('uuids should be either a string or a list')

    if not os.path.exists(save_direc):
        os.makedirs(save_direc)     
        
    endpoint = 'https://api.gdc.cancer.gov/data'
    params = {'ids': uuids}
    response = requests.post(endpoint, 
                             data=json.dumps(params),
                             headers = {"Content-Type": "application/json"})

    response_head_cd = response.headers['Content-Disposition']
    file_name = re.findall('filename=(.+)', response_head_cd)[0]
    file_name = os.path.join(save_direc, file_name)

    with open(file_name, "wb") as output_file:
        output_file.write(response.content)
        
    return file_name

def remove_file(file_name):
    try:
        os.remove(file_name)
    except:
        print('Error deleting file %s' %(file_name))