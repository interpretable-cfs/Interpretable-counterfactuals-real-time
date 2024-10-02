def reparam(mu, std, do_sample=True):
    if do_sample:
        eps = tf.random.normal(tf.shape(std))
        return mu + eps * std
    else:
        return mu

def gaussian_likelihood(x, x_pred):    
    return -tf.reduce_sum(tf.math.squared_difference(x_pred, x))


class DDAE(tf.keras.Model):
    def __init__(self, num_nodes=50, latent_dim=20, op_dim=784, activation_type='relu', num_inf_layers=2, beta1=None, beta2=None, pre_trained=False, adversarial_cls=False,
                 num_gen_layers=3, output_activation_type=None, task='B', categorical_cross_entropy=None, num_classes=10, epsilon1=None, epsilon2=None, num_latents_for_pred=10, epoch_param=1, args=None):
        super(DDAE, self).__init__()

        self.latent_dim = latent_dim
        self.output_activation_type = output_activation_type
        self.num_inf_layers = num_inf_layers
        self.num_gen_layers = num_gen_layers         
        self.adversarial_cls = adversarial_cls
        self.cce = categorical_cross_entropy        
        self.num_classes = num_classes
        self.num_lab_dep_lat = num_latents_for_pred        
        self.pre_trained = pre_trained                              

        if self.latent_dim < self.num_lab_dep_lat:
            raise ValueError(f'Latent used for preds must be < than latent dimensions! Instead {self.num_lab_dep_lat} > {self.latent_dim} was provided!')
        
        self.task = task

        self.model_name = "DDAE"                                    
            self.generative_net = ConvDecNet(latent_dim=latent_dim)
            self.inference_net = ConvEncNet(latent_dim=latent_dim) #determinisitc

        self.mpdv = mean_prior_distribution_variance
        self.z_prior_stdv = tf.constant([1.])
        self.z_prior_stdv_ind = tf.constant([1.])        
        self.z_prior_mean = tf.constant([0.])
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        
        if beta1:
            self.beta1 = beta1
        else:
            self.beta1 = 1
        if beta2:
            self.beta2 = beta2
        else:
            self.beta2 = 1
        if epsilon1:
            self.epsilon1 = epsilon1
        else:
            self.epsilon1 = 1
        if epsilon2:
            self.epsilon2 = epsilon2
        else:
            self.epsilon2 = 1
          
        self.restored_epoch = 0

    def restore_weights(self, task=None, epoch=0):
        checkpoint_dir = self.model_name+'-model-weights'
        if self.cce:
            checkpoint_dir += '_cce'        

        if isinstance(epoch, int):
            inf_str = f'/inf_net_latdim_{self.latent_dim}_epoch_{epoch}'
            gen_str = f'/gen_net_latdim_{self.latent_dim}_epoch_{epoch}'
            self.restored_epoch = epoch + 1
        else:
            raise ValueError('Provide epoch to restore! Epoch value must be an integer!')        
        
        inf_str = f'/{task}_'+inf_str[1:]
        gen_str = f'/{task}_'+gen_str[1:]

        inf_latest_checkpoint = checkpoint_dir+inf_str+'.ckpt' 
        if inf_latest_checkpoint:
            self.inference_net.load_weights(inf_latest_checkpoint) 
            print(f"Inference network weights restored from {inf_latest_checkpoint}")
        else:
            print("No checkpoint found for the inference network.")
        gen_latest_checkpoint = checkpoint_dir+gen_str+'.ckpt' #max( [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if gen_str in f], key=os.path.getctime) #'CDGVAE-model-weights/O_gen_net_epoch_14.ckpt.index'
        if gen_latest_checkpoint:
            self.generative_net.load_weights(gen_latest_checkpoint) #'VAE-model-weights/gen_net_epoch_4.ckpt'
            print(f"Generative network weights restored from {gen_latest_checkpoint}")
        else:
            print("No checkpoint found for the generative network.")

    def get_name(self):
        return self.model_name

    def encode(self, x, eval=False):
        z = self.inference_net(x, eval=eval)
        return z

    def decode(self, z, eval=False):

        if self.output_activation_type is None:
            return self.generative_net(z, eval=eval)
        elif self.output_activation_type == "sigmoid":
            return tf.sigmoid(self.generative_net(z))
        elif self.output_activation_type == "tanh":
            return tf.tanh(self.generative_net(z))
        else:
            raise NotImplementedError("Unsupported output activation type.")


    def call(self, x, labels=None, prior_probs=None, epoch=None):
        latents = self.encode(x) 
        batch_size = latents.shape[0]
        z_ind = latents[:, :self.latent_dim-self.num_lab_dep_lat]
        z_dep = latents[:, self.latent_dim-self.num_lab_dep_lat:]
        cat_label = tf.argmax(labels,axis=1)
        unique_labels, _ = tf.unique(tf.squeeze(cat_label))
        unique_labels = tf.sort(unique_labels)
        num_classes = len(unique_labels)
        if num_classes != self.num_classes:
            raise ValueError('Some label missing in batch! Make sure all labels are present in each batch!')

        grouped_z_ind = [z_ind[cat_label == label] for label in unique_labels]
        moments_l_ind = [tf.nn.moments(z, axes=[0]) for z in grouped_z_ind]
        batch_means_ind = [tf.reshape(moment[0], (1, self.latent_dim-self.num_lab_dep_lat)) for moment in moments_l_ind] #[tf.reshape(moment[0], (1, 5)) for moment in moments_l_ind] #
        batch_vars_ind = [tf.reshape(moment[1], (1, self.latent_dim-self.num_lab_dep_lat)) for moment in moments_l_ind] #[tf.reshape(moment[1], (1, 5)) for moment in moments_l_ind] #
        batch_stds_ind = [tf.tile(tf.math.pow(batch_vars_ind[i], 0.5), [batch_sizes_ind[i], 1]) for i in range(num_classes)]
        batch_sizes_ind = [z.shape[0] for z in grouped_z_ind]        

        grouped_z_dep = [z_dep[cat_label == label] for label in unique_labels]
        moments_l_dep = [tf.nn.moments(z, axes=[0]) for z in grouped_z_dep]
        batch_means_dep = [tf.reshape(moment[0], (1, self.num_lab_dep_lat)) for moment in moments_l_dep]
        batch_vars_dep = [tf.reshape(moment[1], (1, self.num_lab_dep_lat)) for moment in moments_l_dep]
        batch_stds_ind = [tf.tile(tf.math.pow(batch_vars_dep[i], 0.5), [batch_sizes_ind[i], 1]) for i in range(num_classes)]
        batch_sizes_dep = [z.shape[0] for z in grouped_z_dep]
                    
        exponent = -0.5 * tf.reduce_sum(tf.square(tf.expand_dims(tf.cast(z_dep, tf.float64), axis=1) - tf.expand_dims(tf.squeeze(tf.cast(batch_means_dep, tf.float64)), axis=0)), axis=2)
        coefficient = tf.cast(tf.constant(1), dtype=tf.float64) / ((tf.cast(tf.constant(2), dtype=tf.float64) * tf.cast(tf.constant(math.pi), dtype=tf.float64)) ** (tf.cast(num_classes, dtype=tf.float64) / tf.cast(tf.constant(2), dtype=tf.float64)) * tf.sqrt(tf.linalg.det(tf.eye(num_classes, dtype=tf.float64))))
        coefficient = tf.cast(coefficient, tf.float64)
        expon = tf.cast(tf.exp(tf.cast(exponent, tf.float64)), tf.float64)
        probs = coefficient * expon
        if prior_probs is not None:
            probs = tf.multiply(probs, prior_probs)
        probs_selected = tf.reduce_sum(tf.multiply(probs, labels), axis=1)
        labels_predicted = tf.argmax(probs, axis=1)        
        probs_of_pred = tf.reduce_max(probs, axis=1)
        batch_acc = tf.reduce_sum(tf.cast(tf.equal(labels_predicted, cat_label), tf.float32))
        if self.cce:
            row_sums = tf.reduce_sum(probs, axis=1)
            probs_pred_normalized = probs_of_pred / row_sums
            probs_normalized = probs_selected / row_sums
            cat_ce = self.epsilon1*(-tf.reduce_sum(tf.math.log(probs_normalized)))
          
            if contains_nan(row_sums):
                print("row_sums contains NaN values")
            if contains_inf(row_sums):
                print("row_sums contains infinite values")
            if contains_zero(row_sums):
                print("row_sums contains zero values")
              
            if contains_nan(probs_selected):
                print("probs_selected contains NaN values")
            if contains_inf(probs_selected):
                print("probs_selected contains infinite values")
            if contains_zero(probs_selected):
                print("probs_selected contains zero values") 
              
            if contains_nan(probs_normalized):
                print("check contains NaN values")
            if contains_inf(probs_normalized):
                print("check contains infinite values")
            elif contains_zero(probs_normalized):
                print("check contains zero values")

            if contains_nan(cat_ce):
                print("Variable contains NaN values")
                cat_ce = -1000000
            elif contains_inf(cat_ce):
                print("first: Variable contains infinite values")
                cat_ce = -1000000

        if self.adversarial_cls:

            exponent = -0.5 * tf.reduce_sum(tf.square(tf.expand_dims(tf.cast(z_ind, tf.float64), axis=1) - tf.expand_dims(tf.squeeze(tf.cast(batch_means_ind, tf.float64)), axis=0)), axis=2)
            coefficient = tf.cast(tf.constant(1), dtype=tf.float64) / ((tf.cast(tf.constant(2), dtype=tf.float64) * tf.cast(tf.constant(math.pi), dtype=tf.float64)) ** (tf.cast(num_classes, dtype=tf.float64) / tf.cast(tf.constant(2), dtype=tf.float64)) * tf.sqrt(tf.linalg.det(tf.eye(num_classes, dtype=tf.float64))))
            coefficient = tf.cast(coefficient, tf.float64)
            expon = tf.cast(tf.exp(tf.cast(exponent, tf.float64)), tf.float64)
            probs = coefficient * expon
            if prior_probs is not None:
                probs = tf.multiply(probs, prior_probs)
            #print('probs: ', probs.shape)
            probs_selected = tf.reduce_sum(tf.multiply(probs, labels), axis=1)
            labels_predicted = tf.argmax(probs, axis=1)
            probs_of_pred = tf.reduce_max(probs, axis=1)
            batch_acc_adv = tf.reduce_sum(tf.cast(tf.equal(labels_predicted, cat_label), tf.float32))
            if self.cce:
                row_sums = tf.reduce_sum(probs, axis=1)
                probs_pred_normalized = probs_of_pred / tf.reshape(row_sums, (-1,1))
                probs_normalized = probs / tf.reshape(row_sums, (-1,1)) #probs_selected / row_sums
                cat_ce += self.epsilon2*(-tf.reduce_sum(tf.math.log(probs_normalized)/self.num_classes))
              
                if contains_nan(row_sums):
                    print("adv row_sums contains NaN values")
                if contains_inf(row_sums):
                    print("adv row_sums contains infinite values")
                if contains_zero(row_sums):
                    print("adv row_sums contains zero values")
                if contains_nan(probs):
                    print("adv probs contains NaN values")
                if contains_inf(probs):
                    print("adv probs contains infinite values")
                if contains_zero(probs):
                    print("adv probs contains zero values")

                if contains_nan(probs_normalized):
                    print("adv check contains NaN values")
                if contains_inf(probs_normalized):
                    print("adv check contains infinite values")
                elif contains_zero(probs_normalized):
                    print("adv check contains zero values")

                if contains_nan(cat_ce):
                    print("adv Variable contains NaN values")
                    cat_ce = -1000000
                elif contains_inf(cat_ce):
                    print("second: adv Variable contains infinite values")
                    cat_ce = -1000000        
                               
        if self.cce:                    
            if self.adversarial_cls:
                return self.decode(latents), batch_stds_ind, batch_means_ind, batch_stds_dep, batch_means_dep, tf.cast(cat_ce, tf.float32), batch_acc, grouped_z_dep, grouped_z_ind, batch_acc_adv
            else:
                return self.decode(latents), batch_stds_ind, batch_means_ind, batch_stds_dep, batch_means_dep, tf.cast(cat_ce, tf.float32), batch_acc, grouped_z_dep, grouped_z_ind
        else:                                        
            return self.decode(latents), batch_stds_ind, batch_means_ind, batch_stds_dep, batch_means_dep, grouped_z_dep, grouped_z_ind

    def sample_from_gen_model(self, num_samples=25):
        z_samples = tf.random.normal((num_samples, self.latent_dim))
        recons = self.decode(z_samples)
        recons_numpy = recons.numpy().reshape([-1] + [28,28,1])
        f, axarr = plt.subplots(1, num_samples)
        for i in range(num_samples):
            axarr[i].imshow(recons_numpy[i], cmap="gray")
        plt.show()

    def _log_lik_z(self, mu_z, std_z, mu_2, std_2):        
        log_lik = ( std_z ** 2 + (mu_z - mu_2)**2 ) / ( 2 * (std_2 ** 2) ) 
        return tf.reduce_sum(log_lik)

    def neg_elbo(self, x, y=None, prior_probs=None, epoch=None):      
        if self.cce:                    
            if self.adversarial_cls:
                recons, batch_stds_ind, batch_means_ind, batch_stds_dep, batch_means_dep, tf.cast(cat_ce, tf.float32), batch_acc, grouped_z_dep, grouped_z_ind, batch_acc_adv = self.call(x,y,epoch=epoch)
            else:
                recons, batch_stds_ind, batch_means_ind, batch_stds_dep, batch_means_dep, tf.cast(cat_ce, tf.float32), batch_acc, grouped_z_dep, grouped_z_ind = self.call(x,y,epoch=epoch)
        else:                                        
                recons, batch_stds_ind, batch_means_ind, batch_stds_dep, batch_means_dep, grouped_z_dep, grouped_z_ind = self.call(x,y,epoch=epoch)
      
      log_lik_ind = [self._log_lik_z(mu_z=grouped_z_ind[i], std_z=std_mu_z_ind[i], mu_2=self.z_prior_mean, std_2=self.z_prior_stdv) for i in range(self.num_classes)] #[self._kl_divergence_z(mu_z=mean_mu_z_ind[i], std_z=std_mu_z_ind[i], mu_2=self.z_prior_mean, std_2=self.z_prior_stdv) for i in range(self.num_classes)]
      log_lik_dep = [self._log_lik_z(mu_z=grouped_z_dep[i], std_z=std_mu_z_dep[i], mu_2=batch_means_dep[i], std_2=self.z_prior_stdv) for i in range(self.num_classes)]                              
      log_lik_ind = tf.reduce_sum(log_lik_ind, axis=0)      
      log_lik_dep = tf.reduce_sum(log_lik_dep, axis=0)       
      if self.beta1 is not None:
          log_lik_ind = self.beta1 * log_lik_ind
          log_lik_dep = self.beta2 * log_lik_dep
      log_lik_z = log_lik_ind + log_lik_dep

      lik = gaussian_likelihood(x, recon)      
      else:
          if self.cce:                            
              if self.adversarial_cls:                      
                  return - lik + kl_z + cat_ce, lik, batch_acc, batch_acc_adv
              else:
                  return - lik + kl_z + cat_ce, lik, batch_acc
          else:                            
              return - lik + kl_z, lik, batch_acc

    def compute_class_means(self, dataset_obj, p=1):
        labels = []
        latents = []
        num_samples = 1

        for x, y in dataset_obj.next_batch():
            if x.shape[0] > 0:
                if p != 1:
                    samples = np.random.uniform(size=num_samples) < p
                else:
                    samples = [True]

                if samples[0]:                    
                    x = tf.transpose(tf.convert_to_tensor(x), [0, 1, 2, 3])                   
                    latent = self.encode(x)
                    latents.append(latent[:, self.latent_dim-self.num_lab_dep_lat:])
                    labels.append(tf.argmax(y, axis=1))

        concatenated_latents = tf.concat(latents, axis=0)
        concatenated_labels = tf.concat(labels, axis=0)

        unique_labels, idx = np.unique(concatenated_labels, return_inverse=True)
        unique_labels = tf.sort(unique_labels)
        latent_array = tf.squeeze(concatenated_latents)

        grouped_latents = [concatenated_latents[concatenated_labels == label] for label in unique_labels]
        mean_l = [tf.reduce_mean(group, axis=0) for group in grouped_latents]
        concatenated_means = tf.concat(mean_l, axis=0)        
        class_means = tf.reshape(concatenated_means, [self.num_classes, concatenated_latents.shape[1]])
        self.class_means = class_means

        return tf.convert_to_tensor(class_means)

    def compute_class_params(self, dataset_obj, p=1, dependent=True):
        labels = []
        latents = []
        num_samples = 1

        for x, y in dataset_obj.next_batch():
            if p != 1:
                samples = np.random.uniform(size=num_samples) < p
            else:
                if x.shape[0] > 0:
                    samples = [True]
                else:
                    samples = [False]

            if samples[0]:                                  
                x = tf.transpose(tf.convert_to_tensor(x), [0, 1, 2, 3])
                latent = self.encode(x)
                if dependent:
                    latents.append(latent[:, self.latent_dim-self.num_lab_dep_lat:]) #latents.append(latent[:, 5:10]) #
                else:
                    latents.append(latent[:, :self.latent_dim-self.num_lab_dep_lat]) #latents.append(latent[:, :5]) #
                labels.append(tf.argmax(y, axis=1))

        concatenated_latents = tf.concat(latents, axis=0)
        concatenated_labels = tf.concat(labels, axis=0)

        unique_labels, idx = np.unique(concatenated_labels, return_inverse=True)
        unique_labels = tf.sort(unique_labels)
        latent_array = tf.squeeze(concatenated_latents)

        grouped_latents = [concatenated_latents[concatenated_labels == label] for label in unique_labels]
        moments_l = [tf.nn.moments(z, axes=[0]) for z in grouped_latents]
        if dependent:
            mean_l = [tf.reshape(moment[0], (1, self.num_lab_dep_lat)) for moment in moments_l]
            var_l = [tf.reshape(moment[1], (1, self.num_lab_dep_lat)) for moment in moments_l]
        else:
            mean_l = [tf.reshape(moment[0], (1, self.latent_dim-self.num_lab_dep_lat)) for moment in moments_l]
            var_l = [tf.reshape(moment[1], (1, self.latent_dim-self.num_lab_dep_lat)) for moment in moments_l]

        concatenated_means = tf.concat(mean_l, axis=0)
        concatenated_vars = tf.concat(var_l, axis=0)
        class_means = tf.reshape(concatenated_means, [self.num_classes, concatenated_latents.shape[1]])
        class_vars = tf.reshape(concatenated_vars, [self.num_classes, concatenated_latents.shape[1]])
        self.class_means = class_means

        return tf.convert_to_tensor(class_means), tf.convert_to_tensor(class_vars)

    def predict(self, x, class_means=None, dataset_obj=None, p=1, ret_p_z_y=False, prior_probs=None, from_encoding=False):
        if not from_encoding:
            if x.shape[0] > 1:
                x = self.encode(x)
            else:                
                x = self.encode(x)
            x = x = tf.convert_to_tensor(x[:, self.latent_dim-self.num_lab_dep_lat:]) #tf.convert_to_tensor(x[:, 5:10]) #

        if class_means is None:
            if dataset_obj is None:
                raise ValueError('Either provide class-means matrix or a dataset object to compute them!')
            class_means = self.compute_class_means(dataset_obj, p=p)
        mean = class_means

        d = self.num_classes
        if len(mean.shape) == 1:
            mean = tf.expand_dims(mean, 0)
        if len(x.shape) == 1:
            x = tf.expand_dims(x, 0)

        exponent = -0.5 * tf.reduce_sum(tf.square(tf.expand_dims(x, axis=1) - tf.expand_dims(class_means, axis=0)), axis=2) #2
        coefficient = tf.cast(tf.constant(1), dtype=tf.float64) / ((tf.cast(tf.constant(2), dtype=tf.float64) * tf.cast(tf.constant(math.pi), dtype=tf.float64)) ** (tf.cast(d, dtype=tf.float64) / tf.cast(tf.constant(2), dtype=tf.float64)) * tf.sqrt(tf.linalg.det(tf.eye(d, dtype=tf.float64))))
        coefficient = tf.cast(coefficient, tf.float64)
        expon = tf.cast(tf.exp(tf.cast(exponent, dtype=tf.float64)), tf.float64)
        probs = coefficient * expon
        if prior_probs is not None:
            probs = tf.multiply(probs, prior_probs)

        labels_predicted = tf.argmax(probs, axis=1)
        prob_of_pred = tf.reduce_max(probs, axis=1)
        row_sums = tf.reduce_sum(probs, axis=1)
        p_y_z = probs / tf.expand_dims(row_sums, axis=1)

        if ret_p_z_y:
            return labels_predicted, p_y_z, probs
        else:
            return labels_predicted, p_y_z

    def test(self, dataset, class_means=None, p=1, cm_matrix=False):
        if dataset is None:
            raise ValueError('Either provide class-means matrix or a dataset object to compute them!')
        if class_means is None:
            print('COMPUTING MEANS...')
            class_means = self.compute_class_means(dataset, p=p)
            print('DONE!')
        if cm_matrix:
            preds = []
            labs = []

        prior_probs = None #compute_priors(dataset)
        print(prior_probs)
        test_correct = 0.
        test_total = 0.
        # Evaluate model on validation data
        for test_images, test_labels in dataset.next_test_batch(): #next_test_batch
            if test_images.shape[0] > 0:
                test_images = tf.convert_to_tensor(test_images)
                test_labels = tf.convert_to_tensor(test_labels)                
                test_predicted_labels, test_probs = self.predict(tf.transpose(test_images, [0, 1, 2, 3]), class_means, prior_probs=prior_probs)                

                test_true_labels = tf.argmax(test_labels, axis=1)

                if cm_matrix:
                    preds.append(test_predicted_labels)
                    labs.append(test_true_labels)
                else:
                    test_correct += tf.reduce_sum(tf.cast(tf.equal(test_predicted_labels, test_true_labels), tf.float32))
                    test_total += test_images.shape[0]  # Update total count

        if cm_matrix:
            preds = np.concatenate(preds, axis=0)
            labs= np.concatenate(labs, axis=0)
            return preds, labs
        else:
            test_accuracy = test_correct / test_total
            return test_accuracy


class GDAE(tf.keras.Model):
    def __init__(self, num_nodes=50, num_denoise_nodes=64, latent_dim=20, op_dim=784, denoise_lat_dim_dep=8, denoise_lat_dim_ind=3, activation_type='relu', num_inf_layers=2, sigma=0.1, beta1=None, beta2=None, gamma=None,
                 num_gen_layers=3, num_denoise_layers=3, epoch_restore=None, output_activation_type=None, task='B', pre_trained=False, num_latents_for_pred=10, args=None):
        super(GDAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_lab_dep_lat = num_latents_for_pred
        self.denoise_lat_dim_dep = denoise_lat_dim_dep
        self.denoise_lat_dim_ind = denoise_lat_dim_ind
        self.output_activation_type = output_activation_type
        self.num_inf_layers = num_inf_layers
        self.num_gen_layers = num_gen_layers                     
        self.pre_trained = pre_trained
        self.epoch_restore = epoch_restore        
        
        self.task = task
        
        self.model_name = "GDAE"                                                          
        self.generative_net = ConvDecNet(latent_dim=latent_dim)
        self.inference_net = ConvEncNet(latent_dim=latent_dim)                        
        ckp_str_inf = f'CDGBAE-model-weights_cce/{self.task}_inf_net_latdim_{self.latent_dim}_epoch_{self.epoch_restore}.ckpt'
        ckp_str_gen = f'CDGBAE-model-weights_cce/{self.task}_gen_net_latdim_{self.latent_dim}_epoch_{self.epoch_restore}.ckpt'
        self.inference_net.load_weights(ckp_str_inf) 
        self.generative_net.load_weights(ckp_str_gen) 

        for layer in self.inference_net.layers:
            layer.trainable = False       

        self.denoising_enc_dep = FCdenoiseNet(num_nodes=num_denoise_nodes, op_dim=self.denoise_lat_dim_dep, ip_dim=self.num_lab_dep_lat,
                                        activation_type='relu', num_layers=num_denoise_layers)
        self.denoising_dec_dep = FCdenoiseNet(num_nodes=num_denoise_nodes, op_dim=self.num_lab_dep_lat, ip_dim=denoise_lat_dim_dep,
                                        activation_type='relu', num_layers=num_denoise_layers)
        self.denoising_enc_ind = FCdenoiseNet(num_nodes=num_denoise_nodes, op_dim=self.denoise_lat_dim_ind, ip_dim=self.latent_dim-self.num_lab_dep_lat,
                                        activation_type='relu', num_layers=num_denoise_layers)
        self.denoising_dec_ind = FCdenoiseNet(num_nodes=num_denoise_nodes, op_dim=self.latent_dim-self.num_lab_dep_lat, ip_dim=denoise_lat_dim_ind,
                                        activation_type='relu', num_layers=num_denoise_layers)
        
        self.sigma = sigma
        self.z_prior_stdv = tf.constant([1.])
        self.z_prior_mean = tf.constant([0.])
        self.beta1 = beta1
        self.beta2 = beta2        
        self.restored_epoch = 0

    def restore_weights(self, task=None, epoch=0):
        checkpoint_dir = self.model_name+'-model-weights'

        if isinstance(epoch, int):
            denoise_enc_dep_str = f'/den_enc_dep_net_latdim_{self.denoise_lat_dim_dep}_epoch_{epoch}'
            denoise_dec_dep_str = f'/den_dec_dep_net_latdim_{self.denoise_lat_dim_dep}_epoch_{epoch}'
            denoise_enc_ind_str = f'/den_enc_ind_net_latdim_{self.denoise_lat_dim_ind}_epoch_{epoch}'
            denoise_dec_ind_str = f'/den_dec_ind_net_latdim_{self.denoise_lat_dim_ind}_epoch_{epoch}'
            gen_str = f'/mode_iiiii_mpdv{str(self.mpdv*10)}_reg_gen_net_latdim_{self.latent_dim}_epoch_{epoch}'            
            self.restored_epoch = epoch + 1
        else:
            raise ValueError('Provide epoch to restore! Epoch value must be an integer!')
        if task is not None:            
            denoise_enc_dep_str = f'/{task}_'+denoise_enc_dep_str[1:]
            denoise_dec_dep_str = f'/{task}_'+denoise_dec_dep_str[1:]
            denoise_enc_ind_str = f'/{task}_'+denoise_enc_ind_str[1:]
            denoise_dec_ind_str = f'/{task}_'+denoise_dec_ind_str[1:]
            gen_str = f'/{task}_'+gen_str[1:]            
        denoise_enc_dep_latest_checkpoint = checkpoint_dir+denoise_enc_dep_str+'.ckpt' 
        if denoise_enc_dep_latest_checkpoint:
            self.denoising_enc_dep.load_weights(denoise_enc_dep_latest_checkpoint) 
            print(f"DEnoising encoder network weights restored from {denoise_enc_dep_latest_checkpoint}")
        else:
            print("No checkpoint found for the inference network.")
        denoise_dec_dep_latest_checkpoint = checkpoint_dir+denoise_dec_dep_str+'.ckpt' 
        if denoise_dec_dep_latest_checkpoint:
            self.denoising_dec_dep.load_weights(denoise_dec_dep_latest_checkpoint) 
            print(f"Denoising decoder network weights restored from {denoise_dec_dep_latest_checkpoint}")
        else:
            print("No checkpoint found for the inference network.")
        denoise_enc_ind_latest_checkpoint = checkpoint_dir+denoise_enc_ind_str+'.ckpt' 
        if denoise_enc_ind_latest_checkpoint:
            self.denoising_enc_ind.load_weights(denoise_enc_ind_latest_checkpoint) 
            print(f"DEnoising encoder network weights restored from {denoise_enc_ind_latest_checkpoint}")
        else:
            print("No checkpoint found for the inference network.")
        denoise_dec_ind_latest_checkpoint = checkpoint_dir+denoise_dec_ind_str+'.ckpt' 
        if denoise_dec_ind_latest_checkpoint:
            self.denoising_dec_ind.load_weights(denoise_dec_ind_latest_checkpoint) 
            print(f"Denoising decoder network weights restored from {denoise_dec_ind_latest_checkpoint}")
        else:
            print("No checkpoint found for the inference network.")
        gen_latest_checkpoint = checkpoint_dir+gen_str+'.ckpt' 
        if gen_latest_checkpoint:
            self.generative_net.load_weights(gen_latest_checkpoint) 
            print(f"Generative network weights restored from {gen_latest_checkpoint}")
        else:
            print("No checkpoint found for the generative network.")

    def get_name(self):
        return self.model_name

    def encode(self, x, eval=False):
        z = self.inference_net(x, eval)
        return z

    def denoise_encode(self, x, dep=False, eval=False):
        if dep:
            z = self.denoising_enc_dep(x, eval=eval)
        else:
            z = self.denoising_enc_ind(x, eval=eval)
        return z

    def denoise_decode(self,x, dep=False, eval=False):
        if dep:
            return self.denoising_dec_dep(x, eval=eval)
        else:
            return self.denoising_dec_ind(x, eval=eval)

    def decode(self, z, eval=False):
        if self.output_activation_type is None:
            return self.generative_net(z, eval=eval)
        elif self.output_activation_type == "sigmoid":
            return tf.sigmoid(self.generative_net(z))
        elif self.output_activation_type == "tanh":
            return tf.tanh(self.generative_net(z))
        else:
            raise NotImplementedError("Unsupported output activation type.")

    def call(self, x):
        z = self.encode(x) 
        batch_size = z.shape[0]
      
        z_ind = z[:, :self.latent_dim-self.num_lab_dep_lat]
        z_dep = z[:, self.latent_dim-self.num_lab_dep_lat:]        

        noisy_z_dep = reparam(z_dep, self.sigma*tf.ones_like(z_dep), do_sample=True) 
        latents_from_denoiser_dep = self.denoise_encode(noisy_z_dep, dep=True)
        denoised_latents_dep = self.denoise_decode(latents_from_denoiser_dep, dep=True)

        noisy_z_ind = reparam(z_ind, self.sigma*tf.ones_like(z_ind), do_sample=True) 
        latents_from_denoiser_ind = self.denoise_encode(noisy_z_ind, dep=False)
        denoised_latents_ind = self.denoise_decode(latents_from_denoiser_ind, dep=False)
        
        denoised_latents = tf.concat((denoised_latents_ind, denoised_latents_dep), axis=1)                        

        return self.decode(denoised_latents), denoised_latents, z 

    def sample_from_gen_model(self, num_samples=25):
        z_samples = tf.random.normal((num_samples, self.latent_dim))        
        recons = self.decode(z_samples)
        recons_numpy = recons.numpy().reshape([-1] + [28,28,1])
        f, axarr = plt.subplots(1, num_samples)
        for i in range(num_samples):
            axarr[i].imshow(recons_numpy[i], cmap="gray")
        plt.show()
    
    def neg_elbo(self, x, y=None):
      recon, denoised_latents, latents = self.call(x)
      rec_aux = - (self.sigma**2)*( gaussian_likelihood(latents, denoised_latents) )    
      rec = - gaussian_likelihood(x, recon)      
      return rec + rec_aux, rec, kl_z

