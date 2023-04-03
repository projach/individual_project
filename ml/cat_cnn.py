import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

# device configuration to use gpu if there is cuda available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 30
batch_size_training = 48
batch_size_testing = 25
learning_rate = 0.001
num_train_images_per_label = 144
num_eval_images_per_label = 50

data_dir = "D:\study_ml\data_images_v2\cat_breeds"

# Define the transformations you want to apply to your images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
])

# Load the data using ImageFolder
dataset = ImageFolder(root=data_dir, transform=transform)

# Create a dictionary to keep track of the indices of images for each label
label_indices = {}
for idx, (image, label) in enumerate(dataset):
    if label not in label_indices:
        label_indices[label] = []
    label_indices[label].append(idx)

# Create lists of indices for the training and evaluation images
train_indices = []
eval_indices = []
for label, indices in label_indices.items():
    # Select num_train_images_per_label random indices for training
    train_indices += random.sample(indices, num_train_images_per_label)
    # Select num_eval_images_per_label random indices for evaluation
    unused_indices = [idx for idx in indices if idx not in train_indices]
    eval_indices += random.sample(unused_indices, num_eval_images_per_label)

# Create Subsets of your original dataset using the training and evaluation indices
train_dataset = Subset(dataset, train_indices)
eval_dataset = Subset(dataset, eval_indices)
print(
    f'len of training: {len(train_dataset)} len of eval: {len(eval_dataset)}')
# Create a data loader for your dataset
train_loader = DataLoader(
    train_dataset, batch_size=batch_size_training, shuffle=True)
test_loader = DataLoader(
    eval_dataset, batch_size=batch_size_testing, shuffle=True)


# cat categories (breeds)
CATEGORIES = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
              "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx"]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*13*13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# create model(send it to device so to use gpu)
model = ConvNet().to(device)

# using cross entropyLoss because is a multy class classification
criterion = nn.CrossEntropyLoss()
# optimizing model parameters using SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

running_loss = 0.0
for i in range(num_epochs):
    for images, labels in train_loader:
        # Move the data to the device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 10 == 9:    # print every 10 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (num_epochs + 1, i + 1, running_loss / 10))
        running_loss = 0.0
print('finished training')

FILE = "cat_model.pth"
torch.save(model.state_dict(), FILE)

# evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(12)]
    n_class_samples = [0 for i in range(12)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size_training):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy of the network: {acc}%')

    for i in range(12):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'accuracy of {CATEGORIES[i]}: {acc}%')
