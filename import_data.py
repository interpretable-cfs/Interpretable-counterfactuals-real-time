pip install medmnist
import medmnist
from medmnist import BloodMNIST, INFO

class Dataset():
  
    def __init__(self, batch_size=256, test_batch_size=256, dirpath=None, name='B', image_size=None, as_rgb=False):
        trans = transforms.Compose([transforms.ToTensor()])

        self.as_rgb=as_rgb

        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = 'data' #os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data','fmnit_data')
        self.size = 28

        self.name = name
        if self.name in ['B']:        
            data_flag = 'bloodmnist'            
            print(f"Dataset source information : MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
            info = INFO[data_flag]
            print(info['description'])
        elif self.name == 'B':
            if image_size:
                self.size = image_size
            else:
                self.size = 28
            train_set = BloodMNIST(root="./data_train", split="train", transform=trans, download=True, size=self.size, as_rgb=self.as_rgb)
            print(train_set)
            val_set = BloodMNIST(root="./data_val", split="val", transform=trans, download=True, size=self.size, as_rgb=self.as_rgb)
            test_set = BloodMNIST(root="./data_test", split="test", transform=trans, download=True, size=self.size, as_rgb=self.as_rgb)
        
        else:
            raise ValueError(f"User specified name: {self.name} but only acceptable value is 'B'.")
          
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=test_batch_size,
            shuffle=False)
        else:
            self.num_classes = 8

        if self.as_rgb:
            self.data_dims = [self.size, self.size, 3]
        else:
            self.data_dims = [self.size, self.size, 1]
        self.train_size = len(self.train_loader)
        self.val_size = len(self.val_loader)
        self.test_size = len(self.test_loader)
        self.range = [0.0, 1.0]
        self.batch_size = batch_size
        self.num_training_instances = len(train_set)
        self.num_val_instances = len(val_set)
        self.num_test_instances = len(test_set) #len(self.test_loader)

    def next_batch(self):
        for x, y in self.train_loader:       
            if self.as_rgb:
                x = x.permute(0, 2, 3, 1).numpy() #np.reshape(x, (-1, self.size, self.size, 3))
            else:
                x = np.reshape(x, (-1, self.size, self.size, 1))
            y = y.squeeze()
            y_one_hot = np.eye(self.num_classes)[y]
            contains_zero = np.any(np.sum(y_one_hot, axis=0) <= 1)     
          
            if not contains_zero:
                yield x, y_one_hot
            else:
                print('skipping batch because not all classes are present!')
                yield np.array([]), np.array([])


    def next_val_batch(self):
        # one-hot encode???
        for x, y in self.val_loader:
            
            if self.as_rgb:
                x = x.permute(0, 2, 3, 1).numpy() #np.reshape(x, (-1, self.size, self.size, 3))
            else:
                x = np.reshape(x, (-1, self.size, self.size, 1))
            y = y.squeeze()
            y_one_hot = np.eye(self.num_classes)[y]
            contains_zero = np.any(np.sum(y_one_hot, axis=0) <= 1)
          
            if not contains_zero:
                yield x, y_one_hot
            else:
                print('skipping batch because not all classes are present!')
                yield np.array([]), np.array([])

    def next_test_batch(self):
        # one-hot encode???
        for x, y in self.test_loader:
            
            if self.as_rgb:
                x = x.permute(0, 2, 3, 1).numpy() #np.reshape(x, (-1, self.size, self.size, 3))
            else:
                x = np.reshape(x, (-1, self.size, self.size, 1))
            y = y.squeeze()
            y_one_hot = np.eye(self.num_classes)[y]
            contains_zero = np.any(np.sum(y_one_hot, axis=0) <= 1)

            if not contains_zero:
                yield x, y_one_hot
            else:
                print('skipping batch because not all classes are present!')
                yield np.array([]), np.array([])
