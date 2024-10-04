import pickle
import numpy as np
import pandas as pd
import json
import argparse
import math
import seaborn as sns

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

from models import DDAE
from import_data import Dataset

def find_medoids(dataset_obj, model, class_means_dep):

    batch_count = 0
    num_batches = dataset_obj.num_training_instances // dataset_obj.batch_size
    medoids = {'0': None, '1': None, '2': None, '3': None, '4': None, '5': None, '6': None, '7': None}
    diffs = {'0': 1e+6, '1': 1e+6, '2': 1e+6, '3': 1e+6, '4': 1e+6, '5': 1e+6, '6': 1e+6, '7': 1e+6}

    for x, y in dataset_obj.next_batch():
        batch_count += 1
        input_images = x
        input_labels = y
        labels = np.argmax(input_labels, axis=1)

        for i in range(labels.shape[0]):
            instance = tf.convert_to_tensor(np.reshape(x[i], (1,28,28,3)))
            enc_orig = model.encode(instance)
            lab_dep_enc = enc_orig[:, model.latent_dim-model.num_lab_dep_lat:]
            lab_dep_enc = tf.reshape(lab_dep_enc, [model.num_lab_dep_lat])
            squared_differences = tf.square((lab_dep_enc - class_means_dep[labels[i]]))
            euclidean_dist = tf.sqrt(tf.reduce_sum(squared_differences))
            euclidean_dist_value = euclidean_dist.numpy()
            if euclidean_dist_value < diffs[str(labels[i])]:
                diffs[str(labels[i])] = euclidean_dist_value
                medoids[str(labels[i])] = x[i]
        print(f'DONE BTACH {batch_count} OF {num_batches}')
    return medoids


Bdataset_obj = Dataset(batch_size=128, name='B', image_size=28, as_rgb=True)
DDae = DDAE(num_nodes=50, latent_dim=20, op_dim=784, activation_type='relu', num_inf_layers=2, beta1=None, beta2=None, pre_trained=False, adversarial_cls=False,
            num_gen_layers=3, output_activation_type=None, task='B', categorical_cross_entropy=None, num_classes=10, epsilon1=None, epsilon2=None, num_latents_for_pred=10, 
            epoch_param=1, args=None)
DDae.restore_weights(task=Bdataset_obj.name, epoch=int(f'{YOUR EPOCH}')) #INSTERT YUR EPOCH HERE

model = DDae
dataset_obj = Bdataset_obj

medoids = find_medoids(dataset_obj, model, class_means_dep)

medoids_file = f'medoids/{dataset_obj.name}_medoids.json'

with open(medoids_file, 'wb') as f:
    pickle.dump(medoids, f)
with open(medoids_file, 'rb') as f:
    medoids = pickle.load(f)


d = {'0': 'Basophil', '1': 'Eosinophil', '2': 'Erythroblast', '3': 'Immature granulocytes', '4': 'Lymphocyte',
     '5': 'Monocyte', '6': 'Neutrophil', '7': 'Platelet'}
ncols = 4
num_rows = 2
num_classes = DDae.num_classes

f, axarr = plt.subplots(figsize=(4*ncols, 6*num_rows), nrows=num_rows, ncols=ncols)

for i in range(num_classes):
    row = i // 4  
    col = i % 4
    axarr[row, col].imshow(medoids[str(i)], cmap='gray')
    axarr[row, col].set_title(d[str(i)], fontsize=22)
    axarr[row, col].axis('off')

plt.tight_layout()
plt.show()
