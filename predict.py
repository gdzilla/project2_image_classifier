# python menu


# python predict.py ./test_images/orange_dahlia.jpg model.h5
# D:\Gaza\Documents\udacity\intro_ml\deeplearning\project2\p2_image_classifier

# Load libraries

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
#import os

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 
from PIL import Image

# Ignore warnings
import warnings
#warnings.filterwarnings('ignore')

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


import argparse
#Initialize parser
parser = argparse.ArgumentParser()

#Add the image path argument
parser.add_argument('img_path', help= 'Load the image data') 

#Add the model arugment, $$$$$$$$$$$$$44 file_model > my_model
parser.add_argument('model', help='The trained model')

#Add the top_k arugment
parser.add_argument('--top_k', type = int, default = 3, help='Number of K values/classes')

#Add the  category names (classes) argument, $$$$$$$$$$$$$44 label_map > category_names
parser.add_argument('--category_names', default = 'label_map.json', help='Class names') 

# Assign parser to each variable
args = parser.parse_args()

img_path = args.img_path
model = args.model
top_k = args.top_k
category_names = args.category_names

# Print variables
print ("Print image path", img_path)
print("Print model", model)
print("Print top K value", top_k)
print("Print category name", category_names)

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

# Preprocess the image data
#def normalize(img_path, label):

def normalize(image):
    """
     Arg: image, file location

   Returns: 
     img - pre-processed
     
    """
    # Requreiment based on instructions, resize 224x224 pixels 
    img = np.squeeze(image)
    img = tf.image.resize(img, (224, 224)) 
    #image = tf.cast(image, tf.float32)
    img /= 255
    return img

# Predict
def predict(image_path, top_k):
    """
     Arg: image_path, file location
     Arg: top_k, likely class
     Arg: model, reloaded model
     
   Returns: 
     (int) prob 
     (str) classes 
     
    """
    # im = Image.open(image_path)
    img = Image.open(image_path)
    # image = convert image into required format
    img = np.asarray(img)
    img_proc = normalize(img)
    img_proc = np.expand_dims(img_proc, axis=0)
    model = tf.keras.models.load_model('./model.h5', custom_objects={'KerasLayer':hub.KerasLayer})
    # Prediction
    p = model.predict(img_proc)
    classes = np.argpartition(p[0], -top_k)[-top_k:]
    return p[0][classes], (classes+1).astype(str)

x = "-"

# TODO: Plot the input image along with the top 5 classes
print(x*50)
prob, classes = predict(img_path,top_k)
print('Probabilties',prob)
print('Number of classes', classes)
print('Program complete')
print(x*50)

'''
Below is an example of a preferred cmd line method I wanted to do for this project.
However it didn't align with the very specific rubric instructions. 
Could I have done similar to the following and been ok? 


img_data = {'dahlia': './test_images/orange_dahlia.jpg',
      'cautleya': './test_images/cautleya_spicata.jpg'}
      
def load_img_data():
    
    """"
    Returns selected image path
    """
    x = "-"
    print(x*50)
    print(x*50)
    options = input("Ready to predict flower? yes or no") 
    print(x*50)
    print(x*50)
    
    img_path = None
    
    while True:
        if options in ['yes']:
            # Prompt user for flower type
            img = input("Enter flower type:dahlia or cautleya")
            if img in img_data.keys():
            
                # Results in img based on dictionary key
                img_path = img_data[img]
                break
            else:
                print("bad input, try again") 
        else:
            print("Sorry to hear that, take care")
            break
    return img_path
    
'''