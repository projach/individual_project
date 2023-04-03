import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device configuration to use gpu if there is cuda available 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#hyper parameters 
num_epochs = 100
learning_rate = 0.001
num_train_images_per_label = 144
num_eval_images_per_label = 50

data_dir = "D:\study_ml\data_images_v2\cat_breeds"

# Define the transformations you want to apply to your images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
])

train_dataset = torchvision.datasets.CIFAR10(root='./sifar_data',train=True,download=True,transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./sifar_data',train=False,download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_train_images_per_label, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_eval_images_per_label, shuffle=False)

#dataset classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#implement CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        #input is 3 because we have color
        self.conv1 = nn.Conv2d(3, 6, 5)
        #max pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #fully connected layers
        #16*5*5 has to be fixed number and the 10 must be fixed but the other numbers can change
        #we have to put 16*5*5 because after the conv and pool we have a smaller image
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        #first convonutional and pooling layer
        x = self.pool(F.relu(self.conv1(x)))
        #second convonutional and pooling layer
        x = self.pool(F.relu(self.conv2(x)))
        #flaten
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



#create model(send it to device so to use gpu)
model = ConvNet().to(device)

#using cross entropyLoss because is a multy class classification
criterion = nn.CrossEntropyLoss()
#optimizing model parameters using SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
#training loop
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        #origin shape: [4,3,32,32] = 4,3,1024
        #input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)

        #backward and optimize 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 2000 == 0:
            print(f'epoch [{epoch+1}/{num_epochs}], step [{i+1}/{n_total_steps}], loss: {loss.item():.4}')
print('finished training')

#evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        #max returns
        _,predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy of the network: {acc}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'accuracy of {classes[i]}: {acc}%')


