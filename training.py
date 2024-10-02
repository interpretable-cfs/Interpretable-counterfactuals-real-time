from tqdm import tqdm

def train(net, dataset, optimizer='adam', save=False, seed=1020, step_size=0.001, num_epochs=5, save_dir='saved_models'):
    np.random.seed(seed)

    tf.random.set_seed(seed)

    if optimizer.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=step_size)
    elif optimizer.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=step_size)
    else:
        raise ValueError(f"Optimizer {optimizer} not recognized! only 'adam' and 'rmsprop' are implemented.")

    if dataset.name != net.task:
        dic = { 'B': 'blood'}
        raise ValueError(f'Provided wrong dataset: {dic[dataset.name]} for chosen task: {dic[net.task]}!')

    priors = None    
    for epoch in range(num_epochs):

        loss_epoch = 0.
        acc_epoch = 0.
        num_batches = dataset.num_training_instances // dataset.batch_size

        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
            num_usable_batches = 0
            for x, y in dataset.next_batch():

                if x.shape[0] == dataset.batch_size:
                    num_usable_batches += 1                    
                    x = tf.convert_to_tensor(x, dtype=tf.float32)
                    y = tf.convert_to_tensor(y) 

                    with tf.GradientTape() as tape:                        
                        
                        if net.conv_net:                            
                            if net.get_name() == 'CDGRAE':                                                                                                    
                                if net.adversarial_cls:
                                    loss, lik, batch_acc, batch_acc_adv = net.neg_elbo(tf.transpose(x, [0, 1, 2, 3]), y, prior_probs=priors, epoch=epoch)
                                else:
                                    loss, lik, batch_acc = net.neg_elbo(tf.transpose(x, [0, 1, 2, 3]), y, prior_probs=priors, epoch=epoch)  
                             elif net.get_name() == 'DDBAE':
                                    loss, rec, rec2 = net.neg_elbo(tf.reshape(x, [-1, np.prod(dataset.data_dims)]))
                                  
                    gradients = tape.gradient(loss, net.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
                    loss_epoch += loss.numpy()                    
                    acc_epoch += batch_acc.numpy()

                    pbar.update(1)
                    d = {'batch-loss': loss.numpy()}
                    
                    if net.get_name() == 'DGRAE':
                        d['batch-acc'] = batch_acc.numpy()/len(x)
                        if net.adversarial_cls:
                            d['batch-acc-adv'] = batch_acc_adv.numpy()/len(x)
                        d['recon-loss'] = -lik.numpy()                                                            
                    elif net.get_name() == 'DDBAE':
                        d['recon-loss'] = rec.numpy()
                        d['denoising-loss'] = rec2.numpy()
                    pbar.set_postfix(d)                    

            if net.get_name() == 'CDGBAE':                
                print("\nEpoch {} | neg-ELBO {:.4f} | Epoch-ACC {:.4f} ".format(epoch + 1, -loss_epoch / dataset.num_training_instances, acc_epoch / dataset.num_training_instances))
                print("RECOMPUTE MEANS AND EVALUATE MODEL...")
                means, vars = net.compute_class_params(dataset, p=1)
                print(vars)
                print(means)
                print()
                val_correct = 0.
                val_total = 0.
                val_correct_stoch = 0.
                val_total_stoch = 0.
                train_correct = 0.
                train_total = 0.
                train_correct_stoch = 0.
                train_total_stoch = 0.

                for val_images, val_labels in dataset.next_val_batch(): #dataset.next_val_batch():
                    val_images = tf.convert_to_tensor(val_images)
                    val_labels = tf.convert_to_tensor(val_labels)
                    
                    if val_images.shape[0] > 0:                                                     
                        val_predicted_labels, val_probs = net.predict(tf.transpose(val_images, [0, 1, 2, 3]), net.class_means, prior_probs=priors)                        
                        val_true_labels = tf.argmax(val_labels, axis=1)
                        val_correct += tf.reduce_sum(tf.cast(tf.equal(val_predicted_labels, val_true_labels), tf.float32))                        
                        val_total += val_images.shape[0]                      

                for train_images, train_labels in dataset.next_batch():
                    train_images = tf.convert_to_tensor(train_images)
                    train_labels = tf.convert_to_tensor(train_labels)
                  
                    if train_images.shape[0] > 0:
                        
                        train_predicted_labels, train_probs = net.predict(tf.transpose(train_images, [0, 1, 2, 3]), net.class_means, prior_probs=priors)                                                                        
                        train_true_labels = tf.argmax(train_labels, axis=1)
                        train_correct += tf.reduce_sum(tf.cast(tf.equal(train_predicted_labels, train_true_labels), tf.float32))                        
                        train_total += train_images.shape[0]  # Update total count

                val_accuracy = (val_correct / val_total).numpy()
                train_accuracy = (train_correct / train_total).numpy()

                print("| tr-ACC {:.4f} | val_ACC {:.4f}".format(train_accuracy , val_accuracy)) 
                
            else:                                
                print("Epoch {0} | ELBO {1}".format(epoch + 1, -loss_epoch / dataset.num_training_instances))

            if save:
                name = net.get_name()
                save_dir = f"{name}-model-weights"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)                
                if net.get_name() == 'CDGBAE':
                    if net.cce:
                        save_dir += '_cce'                    
                    inf_checkpoint_path = os.path.join(save_dir, f"{dataset.name}_mode_{net.mode}_mpdv{str(net.mpdv*10)}_inf_net_latdim_{net.latent_dim}_epoch_{net.restored_epoch + epoch}.ckpt")
                    gen_checkpoint_path = os.path.join(save_dir, f"{dataset.name}_mode_{net.mode}_mpdv{str(net.mpdv*10)}_gen_net_latdim_{net.latent_dim}_epoch_{net.restored_epoch + epoch}.ckpt")                                
                elif name == 'CDDBAE':
                    denoising_enc_dep_checkpoint_path = os.path.join(save_dir, f"{dataset.name}_mpdv{str(net.mpdv*10)}_den_enc_dep_net_latdim_{net.denoise_lat_dim_dep}_epoch_{net.restored_epoch + epoch}.ckpt")
                    denoising_dec_dep_checkpoint_path = os.path.join(save_dir, f"{dataset.name}_mpdv{str(net.mpdv*10)}_den_dec_dep_net_latdim_{net.denoise_lat_dim_dep}_epoch_{net.restored_epoch + epoch}.ckpt")
                    denoising_enc_ind_checkpoint_path = os.path.join(save_dir, f"{dataset.name}_mpdv{str(net.mpdv*10)}_den_enc_ind_net_latdim_{net.denoise_lat_dim_ind}_epoch_{net.restored_epoch + epoch}.ckpt")
                    denoising_dec_ind_checkpoint_path = os.path.join(save_dir, f"{dataset.name}_mpdv{str(net.mpdv*10)}_den_dec_ind_net_latdim_{net.denoise_lat_dim_ind}_epoch_{net.restored_epoch + epoch}.ckpt")
                    gen_checkpoint_path = os.path.join(save_dir, f"{dataset.name}_mode_iiiii_mpdv{str(net.mpdv*10)}_reg_gen_net_latdim_{net.latent_dim}_epoch_{net.restored_epoch + epoch}.ckpt")                
                
                if name == 'DGVAE' or name == 'CDGVAE':
                    net.inference_net_s.save_weights(inf_s_checkpoint_path)
                    net.inference_net_u.save_weights(inf_u_checkpoint_path)
                    net.generative_net.save_weights(gen_checkpoint_path)
                    print(f"Saved model inference-net_s checkpoint for epoch {epoch} to {save_dir} as {inf_s_checkpoint_path}")
                    print(f"Saved model inference-net_u checkpoint for epoch {epoch} to {save_dir} as {inf_u_checkpoint_path}")
                elif name == 'DDBAE' or name == 'CDDBAE':
                    net.denoising_enc_dep.save_weights(denoising_enc_dep_checkpoint_path)#, save_format='h5')
                    net.denoising_dec_dep.save_weights(denoising_dec_dep_checkpoint_path)#, save_format='h5')
                    net.denoising_enc_ind.save_weights(denoising_enc_ind_checkpoint_path)#, save_format='h5')
                    net.denoising_dec_ind.save_weights(denoising_dec_ind_checkpoint_path)#, save_format='h5')
                    net.generative_net.save_weights(gen_checkpoint_path)#, save_format='h5')
                    print(f"Saved model denoising-encoder-dep-net checkpoint for epoch {epoch} to {save_dir} as {denoising_enc_dep_checkpoint_path}")
                    print(f"Saved model denoising-decoder-dep-net checkpoint for epoch {epoch} to {save_dir} as {denoising_dec_dep_checkpoint_path}")
                    print(f"Saved model denoising-encoder-ind-net checkpoint for epoch {epoch} to {save_dir} as {denoising_enc_ind_checkpoint_path}")
                    print(f"Saved model denoising-decoder-ind-net checkpoint for epoch {epoch} to {save_dir} as {denoising_dec_ind_checkpoint_path}")
                    print(f"Saved model generative-net checkpoint for epoch {epoch} to {save_dir} as {gen_checkpoint_path}\n")                

    print('Training completed!')
