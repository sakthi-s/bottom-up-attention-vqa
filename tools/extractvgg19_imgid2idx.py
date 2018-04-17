"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import cPickle
import numpy as np
import utils



vgg19_train_infile = 'data/img_train.h5'
vgg19_val_infile = 'data/img_val.h5'
vgg19_train_indices_file = 'data/vgg19_train_imgid2idx.pkl'
vgg19_val_indices_file = 'data/vgg19_val_imgid2idx.pkl'



if __name__ == '__main__':

    # VGG19 features
    vgg19_train = h5py.File(vgg19_train_infile, 'r')
    vgg19_val = h5py.File(vgg19_val_infile, 'r')

    train_indices = {}
    val_indices = {}
    for idx, image_id in enumerate(vgg19_train['image_id']):
        train_indices[image_id] = idx;

    for idx, image_id in enumerate(vgg19_val['image_id']):
        val_indices[image_id] = idx;

    cPickle.dump(train_indices, open(vgg19_train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(vgg19_val_indices_file, 'wb'))
    vgg19_val.close()
    vgg19_train.close()
    print("done!")
