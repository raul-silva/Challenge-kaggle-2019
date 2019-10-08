# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:23:08 2019

@author: Raul Alfredo de Sousa Silva
"""
import csv
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import features_extractor as fs

def features(filename1,filename2,address1,address2,featurefile1,featurefile2,lim=3000,a=0,b=0):
    '''
    Given the images located in the 'filenamex', which names are included in 'addressx', this fuction returns all the
    433 features of your dataset into a file with the name given in 'featurefilex'.
    The features are:
        13 features of shape
        348 features of color
        72 features of texture
    To see more details about this features please refer to the report
    Variables:
    # Change this to access training set images
        filename1 = 'XXX'
    # Change this to access test set images
        filename2 = 'XXX'
    # Change this to access training set names
        address1 = 'XXX'
    # Change this to access test set names
        address2 = 'XXX'
    # Change this to rename the training set features file
        featurefile1 = 'XXX'
    # Change this to rename the test set features file
        featurefile2 = 'XXX'
    # A reduction is applied to images in which one of the dimension is greater
    then 3000 pixels by default (to reduce the computational time). You are able
    to change it.
    # Threshold for reduction
    lim = 3000
    # New dimensions in case of reduction
    a=0 size of rows
    b=0 size of columns
    If they were let at 0 the default reduction will be applied, which means
    each dimension divided by 2.
    Syntax:
        features(filename1,filename2,address1,address2,featurefile1,
                 featurefile2,lim=3000,a=0,b=0)
    Observation1: To reduce computational time images with more than 3000 pixels
    are reduced by 2 in the two dimensions
    Observation3: We suppose to have .jpg images
    Observation2: We suppose to have a segmentation map of the image with 
    filename    <image>_segmentation.jpg
    '''
    # Creating labels
    nfeat = 433
    labels = ["ImageID"]
    for i in range(0,nfeat):
        labels.append('f{}'.format(i)) 
    # Loading features of the training-set into a .csv file
    df = pd.read_csv(address1)
    X_df = df['ImageId']
    X_train = X_df.values
    # Creating archive and writing labels
    with open(featurefile1, 'w',newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        # Writing labels
        filewriter.writerow(labels)
    # Writing training-set features in the file
    i=0
    for name_im in X_train:
        filename = filename1+'{}.jpg'.format(name_im)
        image = imread(filename)
        filename_Segmentation = filename1+'{}_segmentation.jpg'.format(name_im)
        image_Segmentation = imread(filename_Segmentation) # Value 0 or 255
        # Use if necessary
        (h,w,c) = image.shape
        if (h > lim or w > lim):
            if (a == 0 or b==0):
                a = int(h/2)
                b = int(w/2)
            h_n = a
            w_n = b
            image = resize(image,(h_n,w_n), mode='reflect')
            image_Segmentation = resize(image_Segmentation,(h_n,w_n), mode='reflect')
            # To get uint8
            seg = (np.round(image_Segmentation)).astype(np.uint8)
            image = (np.round(255*image)).astype(np.uint8)
        else:
            seg = (image_Segmentation/255).astype(np.uint8)
        print()
        features = fs.extract(image,seg,name_im)
        with open(featurefile1, 'a',newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(features)
        i+=1
        print(i,"out of 700")
        
    # Loading features of the trainingset into a .csv file
    df = pd.read_csv(address2)
    X_df = df['ImageId']
    y_df = df['Malignant']
    X_test = X_df.values
    y_test = y_df.values
    
    # Creating labels
    
    with open(featurefile2, 'w',newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        # Writing labels
        filewriter.writerow(labels)
    # Writing features of the test-set into a .csv file
    i=0
    
    for name_im in X_test:
        filename = filename2+'{}.jpg'.format(name_im)
        image = imread(filename)
        filename_Segmentation = filename2+'{}_segmentation.jpg'.format(name_im)
        image_Segmentation = imread(filename_Segmentation) # Value 0 or 255
        # Use if necessary
        (h,w,c) = image.shape
        if (h > lim or w > lim):
            if (a == 0 or b==0):
                a = int(h/2)
                b = int(w/2)
            h_n = a
            w_n = b
            image = resize(image,(h_n,w_n), mode='reflect')
            image_Segmentation = resize(image_Segmentation,(h_n,w_n), mode='reflect')
            # To get uint8
            seg = (np.round(image_Segmentation)).astype(np.uint8)
            image = (np.round(255*image)).astype(np.uint8)
        else:
            seg = (image_Segmentation/255).astype(np.uint8)
        features = fs.extract(image,seg,name_im)
        with open(featurefile2, 'a',newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(features)
        i+=1
        print(i,"out of 300")

