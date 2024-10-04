#### EXPLANATORY PIPELINE REQUIRES A DICTIONARY OF CONCEPTS LIKE THIS: 

d = {
      'basophil':{
                  'concepts':{5:'color', 6:'nucleus shape', 7:'color', 8:'nucleus shape', 9:'nucleus density', 10:'membrane', 11:'shape', 12:'contours', 13:'nucleus size', 14:'size', 15:'size', 16:'nucleus density', 17:'membrane', 18:'mass', 19:'color'},
                  'directions':{5:['less red', 'more red'], 6:['rounder', 'more curved'], 7:['less red', 'more red'], 8:['rounder', 'more curved'], 9:['lighter nucleus', 'havier nucleus'], 10:['bigger', 'smaller'], 11:['more round', 'more oval'], 12:['less polished', 'more polished'], 13:['samller', 'bigger'], 14:['bigger', 'smaller'], 15:['bigger', 'samller'], 16:['less dense', 'more dense'], 17:['bigger', 'smaller'], 18:['less compact', 'more compact'], 19:['more red', 'less red']}
                  },

      'eosinophil':{
                  'concepts':{5:'nucleus darkness', 6:'membrane darkness', 7:'color', 8:'membrane darkness', 9:'nucleus density', 10:'shape', 11:'shape', 12:'nucleus size', 13:'color', 14:'size', 15:'color', 16:'size', 17:'membrane darkness', 18:'nucleus density', 19:'contours'},
                  'directions':{5:['darker', 'lighter'], 6:['darker', 'lighter'], 7:['less red', 'more red'], 8:['darker', 'lighter'], 9:['lighter', 'havier'], 10:['more round', 'less round'], 11:['more round', 'more oval'], 12:['smaller', 'bigger'], 13:['less intense', 'more intense'], 14:['bigger', 'smaller'], 15:['more red', 'less red'], 16:['bigger', 'smaller'], 17:['lighter', 'darker'], 18:['lighter', 'havier'], 19:['more polished', 'less polished']}                
                  },

      'erythroblast':{
                  'concepts':{5:'darkness', 6:'mass', 7:'membrane size', 8:'nucleus shape', 9:'nucleus size', 10:'membrane size', 11:'mass', 12:'size', 13:'membrane size', 14:'nucleus density', 15:'color', 16:'nucleus size', 17:'darkness', 18:'size', 19:'membrane size'},
                  'directions':{5:['darker', 'lighter'], 6:['less compact', 'more compact'], 7:['bigger', 'smaller'], 8:['rounder', 'more curved'], 9:['smaller', 'bigger'], 10:['bigger', 'smaller'], 11:['more compact', 'less compact'], 12:['smaller', 'bigger'], 13:['smaller', 'bigger'], 14:['less', 'more'], 15:['more red', 'less red'], 16:['smaller', 'bigger'], 17:['darker', 'lighter'], 18:['smaller', 'bigger'], 19:['bigger', 'smalle']}
                  },

      'immature granulocytes':{                  
                  'concepts':{5:'color', 6:'nucleus density', 7:'membrane darkness', 8:'nucleus shape', 9:'nucleus size', 10:'membrane darkness', 11:'shape', 12:'size', 13:'color', 14:'nucleus density', 15:'color', 16:'membrane size', 17:'membrane size', 18:'nucleus shape', 19:'contours'},                  
                  'directions':{5:['less red', 'more red'], 6:['less dense', 'more dense'], 7:['lighter', 'darker'], 8:['rounder', 'more curved'], 9:['bigger', 'smaller'], 10:['darker', 'lighter'], 11:['more round', 'more oval'], 12:['smaller', 'bigger'], 13:['less red', 'more red'], 14:['less dense', 'more dense'], 15:['more red', 'less red'], 16:['bigger', 'smaller'], 17:['bigger', 'smaller'], 18:['more curved', 'more round'], 19:['more polished', 'less polished']}                  
                  },

      'lymphocyte':{
                  'concepts':{5:'darkness', 6:'membrane size', 7:'shape', 8:'nucleus shape', 9:'size', 10:'membrane size', 11:'shape', 12:'nucleus size', 13:'membrane size', 14:'denisty', 15:'size', 16:'contours', 17:'nucleus size', 18:'darkness', 19:'membrane size'},
                  'directions':{5:['darker', 'lighter'], 6:['smaller', 'bigger'], 7:['less regular', 'more regular'], 8:['more round', 'more curved'], 9:['smaller', 'bigger'], 10:['bigger', 'smaller'], 11:['more round', 'more oval'], 12:['smaller', 'bigger'], 13:['less membrane', 'more membrane'], 14:['less dense', 'more dense'], 15:['bigger', 'smaller'], 16:['less polished', 'more polished'], 17:['smaller', 'bigger'], 18:['lighter', 'darker'], 19:['bigger', 'smaller']}
                  },                  

      'monocyte':{
                  'concepts':{5:'color', 6:'nucleus size', 7:'nucleus size', 8:'nucleus shape', 9:'nucleus density', 10:'membrane size', 11:'shape', 12:'size', 13:'color', 14:'color', 15:'membrane darkness', 16:'nucleus density', 17:'contours', 18:'shape', 19:'contours'},
                  'directions':{5:['less red', 'more red'], 6:['bigger', 'smaller'], 7:['smaller', 'bigger'], 8:['more round', 'more curved'], 9:['less dense', 'more dense'], 10:['smaller', 'bigger'], 11:['more round', 'more oval'], 12:['smaller', 'bigger'], 13:['less red', 'more red'], 14:['more red', 'less red'], 15:['darker', 'lighter'], 16:['less dense', 'more dense'], 17:['less polished', 'more polished'], 18:['more curved', 'more round'], 19:['more polished', 'less polished']}                  
                  },

      'neutrophil':{
                  'concepts':{5:'nucleus density', 6:'membrane darkness', 7:'color', 8:'nucleus shape', 9:'nucleus density', 10:'mass', 11:'membrane size', 12:'nucleus size', 13:'color', 14:'color', 15:'membrane darkness', 16:'nucleus shape', 17:'contours', 18:'nucleus size', 19:'membrane darkness'},
                  'directions':{5:['more dense', 'less dense'], 6:['more dark', 'less dark'], 7:['less red', 'more red'], 8:['more round', 'more curved'], 9:['less dense', 'more dense'], 10:['less compact', 'more compact'], 11:['bigger', 'smaller'], 12:['smaller', 'bigger'], 13:['less red', 'more red'], 14:['more red', 'less red'], 15:['darker', 'lighter'], 16:['more curved', 'more round'], 17:['less polished', 'more polished'], 18:['smaller', 'bigger'], 19:['more dark', 'less dark']}                  
                  },

      'platelet':{
                  'concepts':{5:'color', 6:'darkness', 7:'darkness', 8:'density', 9:'size', 10:'size', 11:'membrane', 12:'color', 13:'contours', 14:'color', 15:'color', 16:'nucleus size', 17:'darkness', 18:'size', 19:'membrane'},
                  'directions':{5:['less red', 'more red'], 6:['darker', 'lighter'], 7:['darker', 'lighter'], 8:['more dense', 'less dense'], 9:['smaller', 'bigger'], 10:['bigger', 'smaller'], 11:['more membrane', 'less membrane'], 12:['less red', 'more red'], 13:['more polished', 'less polished'], 14:['more red', 'less red'], 15:['more color', 'less color'], 16:['smaller', 'bigger'], 17:['more dark', 'less dark'], 18:['smaller', 'bigger'], 19:['more membrane', 'less membrane']}                  
                  }
    }

#### TO REPLICATE THIS EXPLANATORY TECHNIQUE FIRST EXTRACT CONCEPTS LEARNED BY YOUR MODEL VIA LATENT TRAVERSAL

########################################################################## EXPLANATIONS ####################################################################################

import numpy as np
import pandas as pd
import json
import argparse
import datetime
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, model_from_json, Sequential
from PIL import Image
from tensorflow.keras.layers import Lambda
from tensorflow.keras import losses
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA

import numpy as np
np.set_printoptions(suppress=True)
import math
import time
import random

from models import DDAE, GDAE
from import_data import Dataset

# Record the start time
start_time = time.time()


############ CF GENERATING FUNCTION ############ CF GENERATING FUNCTION  ############ CF GENERATING FUNCTION ############ CF GENERATING FUNCTION



def generate_counterfactuals_ii(adv_class, instance, model, class_means, class_vars, pred):

    if adv_class == pred:
        raise ValueError(f'ASKING A CF OF CLASS {adv_class} WHILE THE MODEL PREIDCTS CLASS {pred}! ONLY CFs OF CLASSES DIFFERENT FROM PREDICTION CAN BE GENERATED!')

    mean1 = class_means[pred]
    mean2 = class_means[adv_class]
    if model.get_name() == 'DGVAE' or model.get_name() == 'CDGVAE':
        enc_orig = model.encode(instance)
        mu_zu = enc_orig[:, :(model.latent_dim-model.num_lab_dep_lat)]
        sigma_zu = tf.math.softplus(enc_orig[:, (model.latent_dim-model.num_lab_dep_lat):(model.latent_dim-model.num_lab_dep_lat)*2])
        mu_zs = enc_orig[:, (model.latent_dim-model.num_lab_dep_lat)*2:(model.latent_dim*2-model.num_lab_dep_lat)]
        sigma_zs = tf.math.softplus(enc_orig[:, (model.latent_dim-model.num_lab_dep_lat)*2+model.num_lab_dep_lat:model.latent_dim*2])
        z_sample_u = reparam(mu_zu, sigma_zu, do_sample=True)
        z_sample_s = reparam(mu_zs, sigma_zs, do_sample=True)
        #enc_orig = np.concatenate([z_sample_u, z_sample_s], axis=1)
        random_point = np.reshape(z_sample_s, (model.num_lab_dep_lat,))
    else:
        random_point = np.reshape(model.encode(instance)[:, model.latent_dim-model.num_lab_dep_lat:], (model.num_lab_dep_lat,))
    n = model.num_lab_dep_lat # 5

    ############ FIRST ROTATION ############

    new_vector1 = np.array(mean1)
    new_vector2 = np.array(random_point)
    new_vector3 = np.array(mean2)
    rotation_matrices = []
    # Compute mean point between adversarial mean and random instance to explain
    M = (mean2 + random_point)/ 2    

    # Zero the last axis
    for i in range(0, n-1):

        angle = np.arctan2(new_vector3[i]-new_vector2[i], new_vector3[i+1]-new_vector2[i+1])

        new_vector1 = new_vector1 - M
        new_vector2 = new_vector2 - M
        new_vector3 = new_vector3 - M

        rotation_matrix = np.eye(n)
        rotation_matrix[i,i] = np.cos(angle)
        rotation_matrix[i,i+1] = -np.sin(angle)
        rotation_matrix[i+1,i] = np.sin(angle)
        rotation_matrix[i+1,i+1] = np.cos(angle)
        rotation_matrices.append(rotation_matrix)

        new_vector1 = np.dot(rotation_matrix, new_vector1) + M
        new_vector2 = np.dot(rotation_matrix, new_vector2) + M
        new_vector3 = np.dot(rotation_matrix, new_vector3) + M

    # Find intercept between decision boundary and segment connecting adversarial mean with instance to explain
    A = new_vector1 #mean1
    B = new_vector3 #mean2
    A = A - new_vector3
    B = B - new_vector3
    AB = A-B # normal vector
    mid = (A+B)/2 # mid-point
    coord = (tf.reduce_sum(mid * AB))/AB[-1] #mid.dot(AB)/AB[-1]
    intersection_point = np.zeros_like(mid)
    intersection_point[-1] = coord    
    print('INTERSECTION:     ', coord) #, nor, AB[-1], mid)

    #### EXPLANATION ####
    std = 1. 
    samples = tf.random.normal(shape=(10,), mean=0, stddev=std) #np.random.normal(0, std, 1000)
    num = 0
    cf = np.zeros_like(mid)
    for el in samples:
        if coord < 0:
            if el > coord and el < 0:
                cf[-1] += el
                num += 1
        else:
            if el < coord and el > 0:
                cf[-1] += el
                num += 1    
    cf[-1] = cf[-1]/num
    print('expected cf1', cf)
    print()
    if to_print:
        print('ROTATED CF:       ', cf)
        print()
    #####################

    # ADD ADVERSARIAL MEAN BEFORE INVERTING ROTATIONS BECAUSE IT WAS ZEROED!
    cf1 = cf + new_vector3
    new_intersection_point = intersection_point + new_vector3

    for rot_matrix in rotation_matrices[::-1]:
        inv_rot_matrix = np.linalg.inv(rot_matrix)
        cf1 = np.dot(inv_rot_matrix, cf1 - M) + M
        new_intersection_point = np.dot(inv_rot_matrix, new_intersection_point - M) + M
        new_vector3 = np.dot(inv_rot_matrix, new_vector3 - M) + M
        new_vector2 = np.dot(inv_rot_matrix, new_vector2 - M) + M
        new_vector1 = np.dot(inv_rot_matrix, new_vector1 - M) + M    

    ############ SECOND ROTATION ############

    new_vector1 = np.array(mean1)
    new_vector2 = np.array(random_point)
    new_vector3 = np.array(mean2)    
    rotation_matrices = []
    # Compute mean point between adversarial mean and predicted class mean
    M = (mean2 + mean1)/ 2
    
    # Zero the x-axis
    for i in range(0, n-1):

        angle = np.arctan2(new_vector3[i]-new_vector1[i], new_vector3[i+1]-new_vector1[i+1])

        new_vector1 = new_vector1 - M
        new_vector2 = new_vector2 - M
        new_vector3 = new_vector3 - M
        #new_intersection_point = new_intersection_point - M

        rotation_matrix = np.eye(n)
        rotation_matrix[i,i] = np.cos(angle)
        rotation_matrix[i,i+1] = -np.sin(angle)
        rotation_matrix[i+1,i] = np.sin(angle)
        rotation_matrix[i+1,i+1] = np.cos(angle)
        rotation_matrices.append(rotation_matrix)

        new_vector1 = np.dot(rotation_matrix, new_vector1) + M
        new_vector2 = np.dot(rotation_matrix, new_vector2) + M
        new_vector3 = np.dot(rotation_matrix, new_vector3) + M
        #new_intersection_point = np.dot(rotation_matrix, new_intersection_point) + M
        
    A = new_vector2 - M # instance to explain    
    
    intersection_point2 = tf.concat((A[:-1], np.array([0])), axis=0)    

    # ADD MEAN-POINT BEFORE INVERTING ROTATIONS BECAUSE IT WAS ZEROED!
  
    new_intersection_point2 = intersection_point2 + M

    for rot_matrix in rotation_matrices[::-1]:
        inv_rot_matrix = np.linalg.inv(rot_matrix)        
        new_intersection_point2 = np.dot(inv_rot_matrix, new_intersection_point2 - M) + M
        new_vector3 = np.dot(inv_rot_matrix, new_vector3 - M) + M
        new_vector2 = np.dot(inv_rot_matrix, new_vector2 - M) + M
        new_vector1 = np.dot(inv_rot_matrix, new_vector1 - M) + M
    
    ############ THIRD ROTATION ############

    new_vector1 = new_intersection_point
    new_vector3 = new_intersection_point2
    new_vectorm1 = np.array(mean1)
    new_vectorm2 = np.array(mean2)
    rotation_matrices = []    
    M = (new_intersection_point + new_intersection_point2)/ 2

    # Zero the x-axis
    for i in range(0, n-1):

        angle = np.arctan2(new_vector3[i]-new_vector1[i], new_vector3[i+1]-new_vector1[i+1])

        new_vector1 = new_vector1 - M
        new_vector3 = new_vector3 - M
        new_vectorm1 = new_vectorm1 - M
        new_vectorm2 = new_vectorm2 - M        

        rotation_matrix = np.eye(n)
        rotation_matrix[i,i] = np.cos(angle)
        rotation_matrix[i,i+1] = -np.sin(angle)
        rotation_matrix[i+1,i] = np.sin(angle)
        rotation_matrix[i+1,i+1] = np.cos(angle)
        rotation_matrices.append(rotation_matrix)

        new_vector1 = np.dot(rotation_matrix, new_vector1) + M
        new_vector3 = np.dot(rotation_matrix, new_vector3) + M
        new_vectorm1 = np.dot(rotation_matrix, new_vectorm1) + M
        new_vectorm2 = np.dot(rotation_matrix, new_vectorm2) + M        

    new_vector3 = new_vector3 - new_vectorm2
    new_vector1 = new_vector1 - new_vectorm2
  
    #### EXPLANATION ####
    std = 1. 
    samples = tf.random.normal(shape=(10,), mean=0, stddev=std) #np.random.normal(0, std, 1000)
    num = 0
    cf = np.zeros_like(new_vector3)
    for el in samples:
        if new_vector3[-1] < new_vector1[-1]:
            if el > new_vector3[-1] and el < new_vector1[-1]:
                cf[-1] += el
                num += 1
        else:
            if el < new_vector3[-1] and el > new_vector1[-1]:
                cf[-1] += el
                num += 1
    print('num2: ', num, new_vector3[-1], new_vector1[-1])
    if num == 0:
        cf[-1] = (new_vector3[-1]+new_vector1[-1])/2
    else:
        cf[-1] = cf[-1]/num

    #####################

    # ADD MEAN-POINT BEFORE INVERTING ROTATIONS BECAUSE IT WAS ZEROED!
    cf2 = cf + tf.concat((new_vector3[:-1], np.array([0])), axis=0)  + new_vectorm2
    new_vector3 = new_vector3 + new_vectorm2
    new_vector1 = new_vector1 + new_vectorm2
    #new_intersection_point2 = intersection_point2 + M

    for rot_matrix in rotation_matrices[::-1]:
        inv_rot_matrix = np.linalg.inv(rot_matrix)
        cf2 = np.dot(inv_rot_matrix, cf2 - M) + M        
        new_vector3 = np.dot(inv_rot_matrix, new_vector3 - M) + M
        new_vectorm2 = np.dot(inv_rot_matrix, new_vectorm2 - M) + M
        new_vectorm1 = np.dot(inv_rot_matrix, new_vectorm1 - M) + M
        new_vector1 = np.dot(inv_rot_matrix, new_vector1 - M) + M

    #################### AVERAGE WITH REALTIVE DENSITIES ####################

    means = class_means[adv_class]
    vars = class_vars[adv_class]
    cov_matrix = tf.linalg.diag(vars)
    normalization = 1 / ((2 * np.pi) ** (len(means) / 2) * np.sqrt(np.linalg.det(cov_matrix)))
    exponent1 = -0.5 * tf.matmul(tf.matmul(tf.transpose(tf.reshape((cf1 - means), (model.num_lab_dep_lat,1))), tf.linalg.inv(cov_matrix)), tf.reshape((cf1 - means), (model.num_lab_dep_lat,1)))
    exponent2 = -0.5 * tf.matmul(tf.matmul(tf.transpose(tf.reshape((cf2 - means), (model.num_lab_dep_lat,1))), tf.linalg.inv(cov_matrix)), tf.reshape((cf2 - means), (model.num_lab_dep_lat,1)))
    dens1 = normalization * np.exp(exponent1)
    dens2 = normalization * np.exp(exponent2)
    weight = dens1/(dens2+dens1)
    cf = (cf1)*(weight) + cf2*(1-weight)

    return cf


############ EXPLANATIONS ############ EXPLANATIONS  ############ EXPLANATIONS ############ EXPLANATIONS ############ EXPLANATIONS ############



def explain_instance(model, id, adv_class, pred, instance, class_means_dep, class_vars_dep, d, to_print=True, lab_ind_enc=None, denoiser=False, class_concepts=None):        

    def remove_duplicates(keys, values):
        unique_association = {}

        # Iterate through the two lists simultaneously
        for key, value in zip(keys, values):
            if key not in unique_association:
                unique_association[key] = value

        # Extract the lists back from the dictionary
        unique_keys = list(unique_association.keys())
        unique_values = list(unique_association.values())

        return unique_keys, unique_values
    # GENERATE COUNTERFACTUAL OF LABEL DEPENDENT ENCODING
    
    cf = generate_counterfactuals_ii(adv_class, instance, model, class_means_dep, class_vars_dep, pred[0])   

    # CREATE NEW INSTANCE WITH LABEL INDEPENDENT ENCODING OF INSTANCE TO EXPLAIN AND LABEL DEPENDENT ENCODING OF CF

    orig_enc = model.encode(instance)
    lab_ind_enc = orig_enc[:, :model.latent_dim-model.num_lab_dep_lat]
    lab_dep_enc = orig_enc[:, model.latent_dim-model.num_lab_dep_lat:]
    lab_dep_enc = tf.reshape(lab_dep_enc, [model.num_lab_dep_lat])    

    # EXTRACT DICTIONARY OF CONCEPTS FOR PREDICTED CLASS
    class_predicted = pred[0]            
    d_concepts = class_concepts[d[f'{adv_class}']]['concepts'] 
    d_directions = class_concepts[d[f'{adv_class}']]['directions']

    #FIND TOP-3 RELEVANT CONCEPTS FOR COUNTERFACTUAL
    squared_differences = tf.square((lab_dep_enc - class_means_dep[class_predicted])/class_vars_dep[class_predicted] ) 
    exponent = -0.5 * squared_differences
    probability_density_orig = (1 / (tf.sqrt(2 * np.pi) * class_vars_dep[class_predicted])) * tf.exp(exponent)             

    squared_differences_cf = tf.square((cf - class_means_dep[adv_class])/class_vars_dep[adv_class] )    
    exponent = -0.5 * squared_differences_cf
    probability_density_cf = (1 / (tf.sqrt(2 * np.pi) * class_vars_dep[adv_class])) * tf.exp(exponent)

    weighted_squared_differences = tf.square(lab_dep_enc*probability_density_orig - cf*probability_density_cf)     
    _, indices = tf.nn.top_k(weighted_squared_differences, k=3)

    # Print the squared differences and the indices of the 3 largest
    print("\nSquared Differences:", weighted_squared_differences.numpy())    
    print("Indices of the 3 largest squared differences:", indices.numpy())        

    lab_dep_enc_numpy = lab_dep_enc.numpy()
    class_means_dep_numpy = class_means_dep[adv_class].numpy()
    cf_numpy = cf.numpy()
    indices_numpy = indices.numpy()     
    directions = [1 if lab_dep_enc_numpy[index]<cf_numpy[index] else 0 for index in indices_numpy] #first value is for increase second for deacrease of latent direction    
    concepts = [d_concepts[f'{indices_numpy[i]+5}'] for i in range(3) ]    
    modes = [d_directions[f'{indices_numpy[i]+5}'][directions[i]] for i in range(3) ]    
    concepts, modes = remove_duplicates(concepts, modes)    

    new_instance = np.concatenate( (np.reshape(lab_ind_enc, (1,model.latent_dim-model.num_lab_dep_lat)), np.reshape(cf, (1,model.num_lab_dep_lat))), axis=1)
    print('\nINSTANCE TO GENERATE: ', new_instance, new_instance.shape)

    # SHOW PREDICTION OF CF INSTANCE

    input = tf.convert_to_tensor(np.reshape(cf, (1,model.num_lab_dep_lat)))    
    pred, prob = model.predict(input, class_means_dep, from_encoding=True)
    print('\nCF PREDICTED CLASS: ', d[f'{pred[0]}'], ' WITH PROBABILITY: ', prob, '\n')

    recons = denoiser.decode(tf.convert_to_tensor(new_instance), eval=True)    
    recons_numpy = recons.numpy().reshape([-1] + [28,28,3])

    return recons_numpy, concepts, modes


def generate_explanations(id, asked_class, model, denoiser, file_name, d):

    ids = []     
    ncols = len(adv_classes) + 1
    dataset_obj = Bdataset_obj

    with open(file_name, 'r') as json_file:
        class_concepts = json.load(json_file)
    
    if isinstance(id, int): 
        ids.append(id)
        for x, y in dataset_obj.next_test_batch(): 
            input_images = x
            input_labels = y
            break
    else:
        for x, y in dataset_obj.next_test_batch(): 
            input_images = x
            input_labels = y
            if asked_class in [0, 1, 2, 3, 4, 5, 6, 7]:
                ids = []
                labels = np.argmax(input_labels, axis=1)
                for i in range(labels.shape[0]):
                    if labels[i] == asked_class:                                                
                        ids.append(i)
                random.shuffle(ids)
            else:
                ids = random.sample(range(128), num_samples)
            break

    print(f'ID: {ids[0]}')

    # PLOT COUNTERFACTUALS FOR EACH ASKED CLASS
    f, axarr = plt.subplots(figsize=(size*ncols, size), nrows=1, ncols=ncols)

    # RECONSTRUCT ORIGINAL INSTANCE
    recons_orig = model.decode(enc_orig)
    recons_orig_numpy = recons_orig.numpy().reshape([-1] + [28,28,3])                        
    axarr[0].imshow(instance[0], cmap="gray")
    axarr[0].set_title(f'Model prediction: ' + f'{d[str(pred[0].numpy())]}'.capitalize(), fontsize=35, fontweight='bold') 
    axarr[0].axis('off')            
    # EXPLAIN INSTANCE WITH COUNTERFACTUALS
    for j in range(len(adv_classes)):
        if adv_classes[j] != pred[0]:
            recons_numpy, concepts, modes = explain_instance(model=model, id=ids[0], adv_class=adv_classes[j], pred=pred, instance=instance, denoiser=denoiser, d,
                                                                                        class_means_dep=class_means_dep, class_vars_dep=class_vars_dep, lab_ind_enc=z_sample_u, class_concepts=class_concepts)             
            axarr[j+1].imshow(recons_numpy[0], cmap="gray")
            axarr[j+1].set_title(f'Counter example for class ' + f'{d[str(adv_classes[j])]}'.capitalize(), fontsize=35, fontweight='bold')                  
            y_pos = 27.7  # Start position for the first concept
            for i in range(len(concepts)):
                axarr[j+1].text(0, y_pos, f'{concepts[i]}: '.capitalize() + f'{modes[i]}'.capitalize(), fontsize=30, fontweight='bold', verticalalignment='top', color='black', wrap=True)
                y_pos += 1.5
            axarr[j+1].axis('off')
        else:              
            black_image = np.zeros((28, 28))
            axarr[j+1].imshow(black_image, cmap='gray')
            axarr[j+1].set_title(f'Predicted class slot. No CF returned', fontsize=25, fontweight='bold')
            axarr[j+1].axis('off')
        
    plt.subplots_adjust(bottom=-0.35, top=0.3, wspace=0.3)
    plt.tight_layout()        

    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")    


# CREATE MODEL INSTANCE, SELECT DATASET AND PICK TEST INSTANCE TO EXPLAIN

###### EXECUTE ONLY ONCE!!!
Bdataset_obj = Dataset(batch_size=128, name='B', image_size=28, as_rgb=True)

DDae = DDAE(num_nodes=50, latent_dim=20, op_dim=784, activation_type='relu', num_inf_layers=2, beta1=None, beta2=None, pre_trained=False, adversarial_cls=False,
            num_gen_layers=3, output_activation_type=None, task='B', categorical_cross_entropy=None, num_classes=10, epsilon1=None, epsilon2=None, num_latents_for_pred=10, 
            epoch_param=1, args=None)

GDae = GDAE(num_nodes=64, num_denoise_nodes=64, latent_dim=20, op_dim=784, denoise_lat_dim_=12, activation_type='relu', num_inf_layers=2, sigma=0.1, 
            beta1=None, beta2=None, gamma=None,num_gen_layers=3, num_denoise_layers=3, epoch_restore=None, conv_net=False, output_activation_type=None, task='B', 
            pre_trained=False, num_latents_for_pred=10, args=None)

DDae.restore_weights(task=Bdataset_obj.name, epoch=f'{YOUR EPOCH}') #INSTERT YUR EPOCH HERE
GDae.restore_weights(task=Bdataset_obj.name, epoch=f'{YOUR EPOCH}') #INSERT YOUR EPOCH HERE
class_means_dep, class_vars_dep = DDae.compute_class_params(Bdataset_obj, p=1, dependent=True)
class_means_ind, class_vars_ind = DDae.compute_class_params(Bdataset_obj, p=1, dependent=False)
########


id = None 
asked_class = random.sample(range(model.num_classes), 1)[0]
model = DDae
denoiser = GDae
adv_classes_main = list(range(model.num_classes)) 
size = 11 
file_name = 'class_concepts_and_directions2.json' 
d = {'0': 'basophil', '1': 'eosinophil', '2': 'erythroblast', '3': 'immature granulocytes', '4': 'lymphocyte', '5': 'monocyte', '6': 'neutrophil', '7': 'platelet'}

generate_explanations(id, asked_class, model, denoiser, file_name, d, size=size)                       
                       
                       
