# library of functions to prepare image data for machine learning
# functions created by Emmett Culhane, copied from cnn-12V1.ipynb and with docstrings and some changes added here

import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
import os, re
import cv2
import imageio

def preprocess_input(image, fixed_size=128):
    '''
    
    '''
    
    image_size = image.shape[:2] 
    ratio = float(fixed_size)/max(image_size)
    new_size = tuple([int(x*ratio) for x in image_size])
    img = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = fixed_size - new_size[1]
    delta_h = fixed_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    ri = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #print(ri.shape)
    #gray_image = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
    #gimg = np.array(gray_image).reshape(fixed_size,fixed_size,1)
    gimg = np.array(ri).reshape(fixed_size,fixed_size,1)
    img_n = cv2.normalize(gimg, gimg, 0, 255, cv2.NORM_MINMAX)
    return(img_n)


# build a list of PNGs that match the rows of subsampled metadata from image-file-directory.csv
# this function from AC and VS
def buildPNGsName(imageid):
    '''
    Build a list of PNGs that match the rows of subsampled metadata
    Inputs
    ------
      imageid (str):
        example format: IFCB107D20151109T221417P00789
      
    Outputs
    -------
      png_name (str)
        name of the png and the folder it belongs to (relative path). 
        example format: D20151128T170950_IFCB107/IFCB107D20151128T170950P00079.png
        The folder name and the png name should match with exception of the underscore and the unique image id (PXXXXX)
    '''
    file_name = imageid + '.png'
    datetime = imageid[-22:-6]
    instrument_name = imageid[:-22]
    folder_name = datetime + '_' + instrument_name
    png_name = os.path.join(folder_name,file_name)
    return png_name


# this came from multi_stream_generator_SLC
def image_generator(dataset, batch_size, lb):
    '''
    '''
    data_size = len(dataset)
    n_batches = data_size / batch_size
    remain = data_size % batch_size 
    while True: 
        files = dataset.sample(n=data_size - remain)
        shuffled = files.sample(frac=1)
        result = np.array_split(shuffled, n_batches)  
        for batch in result: 
            labels = batch['high_group'].values
            labels = lb.transform(labels)
            image_data = []
            for i in range(len(batch)): 
                row = batch.iloc[i]
                input_path = row['full_path']
                #image_data.append(preprocess_input(cv2.imread(input_path)))
                image_data.append(preprocess_input(imageio.imread(input_path)))
            image_data = np.array(image_data)
            yield (image_data, labels )
