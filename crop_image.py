# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:15:00 2019

@author: raul-
"""
import numpy as np
from skimage.transform import resize
from skimage.io import imread,imsave

def crop_image(img, seg, size):
    '''
		To crop images with this function you should give the following data:
		img: the image itself
		seg: the segmentation chart of the image
		size: the desired size of the image in the format [a,b]
	'''
    (m,n) = np.shape(seg)
    
    ind = np.where(seg == 1)
    
    dim = max((np.max(ind[0]) - np.min(ind[0])),(np.max(ind[1]) - np.min(ind[1])))
    
    imin = (np.max(ind[0]) + np.min(ind[0]))/2 - dim/2
    imax = (np.max(ind[0]) + np.min(ind[0]))/2 + dim/2
    jmin = (np.max(ind[1]) + np.min(ind[1]))/2 - dim/2
    jmax = (np.max(ind[1]) + np.min(ind[1]))/2 + dim/2
    
    imin = int(max(0,imin-50))
    imax = int(min(m,imax+50))
    jmin = int(max(0,jmin-50))
    jmax = int(min(n,jmax+50))
    
    crp_i = img[imin:imax,jmin:jmax,:]
    crp_s = seg[imin:imax,jmin:jmax]
    
    h_n = size[0]
    w_n = size[1]
    cropped_img = (255*resize(crp_i,(h_n,w_n), mode='reflect')).astype(np.uint8)
    cropped_seg = (255*resize(crp_s,(h_n,w_n), mode='reflect')).astype(np.uint8)
    
    return cropped_img, cropped_seg
#%%
df = pd.read_csv('C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\train.csv')
X_df = df['ImageId']
X_train = X_df.values

for name_im in X_train[1:2]:
    filename = 'C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\im\\{}.jpg'.format(name_im)
    image = imread(filename)
    filename_Segmentation = 'C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\im\\{}_segmentation.jpg'.format(name_im)
    image_Seg = imread(filename_Segmentation) # Value 0 or 255
    size = [256,256]
    img, seg = crop_image(image,image_Seg,size)
    nome_i = "images/"+name_im+".jpg"
    nome_s = "images/"+name_im+"_seg.jpg"
    imsave(nome_i,img)
    imsave(nome_s,seg)
    print("Hello world")

df = pd.read_csv('C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\test.csv')
X_df = df['ImageId']
X_test = X_df.values
for name_im in X_test[1:2]:
    filename = 'C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\im\\{}.jpg'.format(name_im)
    image = imread(filename)
    filename_Segmentation = 'C:\\Users\\raul-\\Documents\\DataSet-Challenge-IMA205-2019\\data\\im\\{}_segmentation.jpg'.format(name_im)
    image_Seg = imread(filename_Segmentation) # Value 0 or 255
    size = [256,256]
    img, seg = crop_image(image,image_Seg,size)
    nome_i = "images/"+name_im+".jpg"
    nome_s = "images/"+name_im+"_seg.jpg"
    imsave(nome_i,img)
    imsave(nome_s,seg)
    print("Hello world")
