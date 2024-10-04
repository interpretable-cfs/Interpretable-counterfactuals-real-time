import pickle

def find_medoids(dataset_obj, model, class_means_dep):

    batch_count = 0
    num_batches = dataset_obj.num_training_instances // dataset_obj.batch_size
    medoids = {'0': None, '1': None, '2': None, '3': None, '4': None, '5': None, '6': None, '7': None}
    diffs = {'0': 1e+6, '1': 1e+6, '2': 1e+6, '3': 1e+6, '4': 1e+6, '5': 1e+6, '6': 1e+6, '7': 1e+6}

    for x, y in dataset_obj.next_batch(): #next_test_batch
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
