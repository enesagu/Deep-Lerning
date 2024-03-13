#%%
!pip install torch
!pip imstall pillow
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device ",device)

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

x_train_negative_tensor = torch.from_numpy(train_negative_array)
print("x_train_negative_tensor:",x_train_negative_tensor.size())

y_train_negative_tensor = torch.zeros(num_train_negative_img, dtype=torch.long)
print("y_train_negative_tensor:",y_train_negative_tensor.size())

#   read train positive
train_positive_path = r"/home/tractus/Desktop/example/python/python1/LSIFIR/Classification/Train/pos"
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path,num_train_positive_img)
x_train_positive_tensor = torch.from_numpy(train_positive_array)
print("x_train_positive_tensor:",x_train_positive_tensor.size())

y_train_positive_tensor = torch.ones(num_train_positive_img, dtype=torch.long)
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

print(x_train)
print(y_train)


x_test = torch.cat((x_test_negative_tensor,x_test_positive_tensor),0)
y_test = torch.cat((y_test_negative_tensor,y_test_positive_tensor),0)

print(x_test)
print(y_test)
#%% deneme
plt.imshow(x_train[42001,:].reshape(64,32),cmap="gray")


#%% cnn

# hyperparameter

num_epochs = 5000
num_classes = 2
batch_size = 8933
learning_rate = 0.00001


class Net(nn.Module):
    
    def __init__(self):
        super(Net.self).__init__()

        self.conv1 = nn.Conv2d(1,10,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,16,5)

        self.fc1 = nn.Linear(16*13,*5,520)
        self.fc2 = nn.Linear(520*130)
        self.fc3 = nn.Linear(130, num_classes)


        def forward(self,x):
            x = self.pool(F.relu((self.conv1(x))))
            x = self.pool(F.relu(self.conv2(x)))

            x = x.view(-1,16*13*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x



train = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.Dataloader(train,batch_size = batch_size, shuffle = True)
 

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.Dataloader(test, batch_size = batch_size, shuffle = True)
 

net = Net()

#%%
criterion = nn.CrossEntropyLoss

import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr=learning_rate, momentum=0.8)

#%% train a network

start = time.time()
train_acc = []
test_acc = []
loss_list = []

use_gpu = False # True


for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        
        inputs,labels = data
        inputs = inputs.view(batch_size,1,64,32) # reshape
        inputs = inputs.float() #float


        # use gpu
        if use_gpu:
            if torch.cuda.is_available():
                inputs,labels=inputs.to(device), labels.to(device)

        # zero gradient
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)

        # loss
        loss = criterion(outputs,labels)

        # back
        loss.backward()
        optimizer.step()


    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(batch_size,1,64,32)



# train











end = time.time()
procces_time = (end-start)/60
print("Procces time: ",procces_time)

























# %%