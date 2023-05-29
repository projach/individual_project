import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import train_models
from torch.utils.data import Dataset
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torchinfo import summary

# device configuration to use gpu if there is cuda available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 50
batch_size_training = 48
batch_size_testing = 25
learning_rate = 0.001
num_train_images_per_label = 144
num_eval_images_per_label = 50

data_dir = "D:\individual_project\data_images_v2\cat_breeds"

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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 28 * 28, 12)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
        
# create model(send it to device so to use gpu)
model = ConvNet().to(device)

# summary(model = model,
#         input_size=(batch_size_testing, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )

# using cross entropyLoss because is a multy class classification
loss_fn = nn.CrossEntropyLoss()
# optimizing model parameters using SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

train_models.train(model=model,train_dataloader=train_loader,test_dataloader=test_loader,
                   optimizer=optimizer,loss_fn=loss_fn,epochs=num_epochs,device=device)

# FILE = "cat_model.pth"
# torch.save(model.state_dict(), FILE)

