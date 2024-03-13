#%%
!pip install torch
!pip install pillow
!pip install matplotlib
!pip install numpy
import torch
import torch.nn as nn
import torch.nn.functional as F 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import os
import time
import torch.utils.data

#%% Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device ",device)

#%% Dataset

def read_images(path, num_img):
    array = np.zeros([num_img,64*32])

    i = 0

    for img in os.listdir(path):
        img_path = os.path.join(path, img)  # Dosya yollarını birleştirmek için os.path.join() fonksiyonunu kullanın
        img = Image.open(img_path, mode='r')
        data = np.asarray(img, dtype="uint8")
        data = data.flatten()
        array[i,:] = data
        i += 1
    return array


#   read train negative
train_negative_path = r"/home/tractus/Desktop/example/python/python1/LSIFIR/Classification/Train/neg"
num_train_negative_img = 43390

train_negative_array = read_images(train_negative_path,num_train_negative_img)

x_train_negative_tensor = torch.from_numpy(train_negative_array[:40000,:])
print("x_train_negative_tensor:",x_train_negative_tensor.size())

y_train_negative_tensor = torch.zeros(42000, dtype=torch.long)
print("y_train_negative_tensor:",y_train_negative_tensor.size())

#   read train positive
train_positive_path = r"/home/tractus/Desktop/example/python/python1/LSIFIR/Classification/Train/pos"
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path,num_train_positive_img)
x_train_positive_tensor = torch.from_numpy(train_positive_array[:10000,:])
print("x_train_positive_tensor:",x_train_positive_tensor.size())

y_train_positive_tensor = torch.ones(10000, dtype=torch.long)
print("y_train_positive_tensor:",y_train_positive_tensor.size())

# read test negative 22050
test_negative_path = r"/home/tractus/Desktop/example/python/python1/LSIFIR/Classification/Test/neg"
num_test_negative_img = 22050
test_negative_array = read_images(test_negative_path,num_test_negative_img)
x_test_negative_tensor = torch.from_numpy(test_negative_array[:20855,:])
print("x_text_negative_tensor:",x_test_negative_tensor.size())
y_test_negative_tensor = torch.zeros(20855, dtype = torch.long)
print("y_test_negative_tensor:",y_test_negative_tensor.size())

# read test positive 5944
test_positive_path = r"/home/tractus/Desktop/example/python/python1/LSIFIR/Classification/Test/pos"
num_positive_img = 22050
test_positive_array = read_images(test_positive_path,num_positive_img)
x_test_positive_tensor = torch.from_numpy(test_positive_array)
print("x_text_positive_tensor:",x_test_positive_tensor.size())
y_test_positive_tensor = torch.zeros(num_test_negative_img, dtype = torch.long)
print("y_test_positive_tensor:",y_test_negative_tensor.size())

# concat train
x_train = torch.cat((x_train_negative_tensor,x_train_positive_tensor),0)
y_train = torch.cat((y_train_negative_tensor,y_train_positive_tensor),0)

print(x_train.size())
print(y_train.size())

# concat test
x_test = torch.cat((x_test_negative_tensor,x_test_positive_tensor),0)
y_test = torch.cat((y_test_negative_tensor,y_test_positive_tensor),0)

print(x_test.size())
print(y_test.size())
print("read section")
#%% deneme
plt.imshow(x_train[41999,:].reshape(64,32),cmap="gray")

#%%

# hyperparameter

num_epochs = 100
num_classes = 2
batch_size = 2000
learning_rate = 0.00001

train = torch.utils.data.Dataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(train,batch_size=batch_size, shuffle=True)

test = torch.utils.data.Dataset(x_test,y_test)
trainloader = torch.utils.data.DataLoader(test,batch_size=batch_size, shuffle=False)


