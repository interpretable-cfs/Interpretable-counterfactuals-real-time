import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from models import GDAE
from import_data import Dataset

def explain(net, input_images, edit_dim_id, edit_dim_value, edit_z_sample=False):

    reference_z = net.encode(input_images, eval=True)
    edited_z = reference_z.numpy().copy()
    edited_z[:, edit_dim_id] = edit_dim_value            
    
    latents_from_denoiser = net.denoise_encode(edited_z, dep=True, eval=True) 
    denoised_latents_dep = net.denoise_decode(latents_from_denoiser, dep=True, eval=True)    
    edited_images = net.decode(edited_z, eval=True)

    return edited_images


def plot_latent_traversal(dataset_obj, net, input_example=None, input_images=None, num_sweeps=15,
                          max_abs_edit_value=10.0, epoch=0, batch_id=0):

    l = [-1, dataset_obj.size, dataset_obj.size, 3]        
    param = net.latent_dim
    param2 = 0                
    reference_z = net.encode(tf.reshape(input_example, l), eval=True)
    print('original encoding: ', reference_z)

    edit_dim_values = np.linspace(-1.0 * max_abs_edit_value, max_abs_edit_value, num_sweeps)

    f, axarr = plt.subplots(param, len(edit_dim_values), sharex=True, sharey=True)
    f.set_size_inches(15, 15 * param / len(edit_dim_values))

    for i in range(param):
        for j in range(len(edit_dim_values)):                                    
            edited_image = convert_and_reshape(explain(net, input_images=tf.reshape(input_example, l),
                                                        edit_dim_id=param2+i, edit_dim_value=edit_dim_values[j], edit_z_sample=False), dataset_obj)

            if edited_image.shape[3] == 1:
                axarr[i][j].imshow(edited_image[0], cmap="gray", aspect='auto') 
            else:
                axarr[i][j].imshow(edited_image[0] * 0.5 + 0.5, cmap="gray", aspect='auto')
            if i == len(axarr) - 1:
                axarr[i][j].set_xlabel("z:" + str(np.round(edit_dim_values[j], 1)))
            if j == 0:
                axarr[i][j].set_ylabel("l:" + str(i))
            axarr[i][j].set_yticks([])
            axarr[i][j].set_xticks([])

    plt.subplots_adjust(hspace=0, wspace=0)

    plt.show()



d = {'0': 'basophil', '1': 'eosinophil', '2': 'erythroblast', '3': 'immature granulocytes', '4': 'lymphocyte',
     '5': 'monocyte', '6': 'neutrophil', '7': 'platelet'}
id = 0
class_to_plot = 2 #CHOOSE CLASS
medoids_file = f'medoids/B_medoids.json'
with open(medoids_file, 'rb') as f:
    medoids = pickle.load(f)
plot_latent_traversal(Bdataset_obj, GDae, max_abs_edit_value=10, input_example=medoids[str(class_to_plot)], input_images=None)
