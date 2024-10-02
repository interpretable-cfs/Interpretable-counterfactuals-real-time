
class ConvEncNet(tf.keras.Model):
    def __init__(self, latent_dim=20, num_channels=1, activation_type='relu', task='D'):
        super(ConvEncNet, self).__init__()
        self.task = task

        self.activation_type = activation_type        

        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='valid') 
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='valid') 
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='valid') 
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='valid') 
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(200) 
        self.fc2 = tf.keras.layers.Dense(200) 
        self.fc3 = tf.keras.layers.Dense(latent_dim) 
        
        if activation_type == 'relu':
            self.activation = tf.keras.layers.ReLU()
        elif activation_type == 'tanh':
            self.activation = tf.keras.layers.Activation('tanh')
        elif activation_type == 'leaky_relu':
            self.activation = tf.keras.layers.ReLU(negative_slope=0.2)
        else:
            raise ValueError("Activation Type not supported")

    def call(self, x, eval=False):
        
        x = self.conv1(x) #28 -> 26
        x = self.batch_norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x) #26 -> 24
        x = self.batch_norm2(x)
        x = self.activation(x)
        
        x = self.pool(x) #24 -> 12
        
        x = self.conv3(x) #12 -> 10
        x = self.batch_norm3(x)
        x = self.activation(x)
        
        x = self.conv4(x) #10 -> 8
        x = self.batch_norm4(x)
        x = self.activation(x)
        
        x = self.flatten(x) #8x8 = 64
        
        x = self.fc1(x) #200
        x = self.activation(x)
        #x = self.dropout(x)
        
        x = self.fc2(x) #200
        x = self.activation(x)
        
        x = self.fc3(x) #latent_dim        
        return x


class ConvDecNet(tf.keras.Model):
    def __init__(self, latent_dim=10, num_channels=3, activation_type='relu', task='D'):
        super(ConvDecNet, self).__init__()
        self.task = task

        self.activation_type = activation_type

        self.fc1 = tf.keras.layers.Dense(200) 
        self.fc2 = tf.keras.layers.Dense(200)      
        self.f38 = tf.keras.layers.Dense(8*8*64)       
        self.conv1 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='valid', use_bias=True)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='valid', use_bias=True)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv3 = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding='valid', use_bias=True)
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2DTranspose(num_channels, kernel_size=3, strides=1, padding='valid', use_bias=True)
        self.batch_norm4 = tf.keras.layers.BatchNormalization()

        if activation_type == 'relu':
            self.activation = tf.keras.layers.ReLU()
        elif activation_type == 'tanh':
            self.activation = tf.keras.layers.Activation('tanh')
        else:
            raise ValueError("Activation Type not supported")

    def call(self, x, eval=False):
        
        x = self.fc1(x) 
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        
        x = tf.reshape(x, (-1, 8, 8, 64)) #64
        
        x = self.conv1(x) #8 -> 10
        x = self.batch_norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x) #10 -> 12
        x = self.batch_norm2(x)
        x = self.activation(x)
        
        x = self.upsample(x) #12 -> 24
        
        x = self.conv3(x) #24 -> 26
        x = self.batch_norm3(x)
        x = self.activation(x)
        
        x = self.conv4(x) #26 -> 28      
        return x


class FCdenoiseNet(tf.keras.Model):
    def __init__(self, num_nodes=50, ip_dim=1, op_dim=1, activation_type='relu', num_layers=2):
        super(FCdenoiseNet, self).__init__()

        self.num_layers = num_layers
        if activation_type == 'relu':
            self.activation = tf.keras.layers.ReLU()
        elif activation_type == 'tanh':
            self.activation = tf.keras.layers.Activation('tanh')
        else:
            raise ValueError("Activation Type not supported")
        self.fc1 = tf.keras.layers.Dense(num_nodes)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc_hidden = []
        for _ in range(self.num_layers - 1):
            num_nodes = num_nodes/2
            self.fc_hidden.append(tf.keras.layers.Dense(num_nodes))           
            self.fc_hidden.append(self.activation)
        self.features = tf.keras.Sequential(self.fc_hidden)
        self.fc_out = tf.keras.layers.Dense(op_dim)

    def call(self, x, eval=False):
        if not eval:
            x = tf.squeeze(x)
        x = self.fc1(x)        
        x = self.activation(x)
        x = self.features(x)
        x = self.fc_out(x)
        return x
