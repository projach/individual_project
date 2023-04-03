from timeit import default_timer as timer
import matplotlib.pyplot as plt
import torch
import torchvision
import random
import torchvision.transforms as transforms
import numpy as np
import train_models
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset


# to utilise gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
batch_size_training = 48
batch_size_testing = 25


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

weights = torchvision.models.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights).to(device)

# freeze layers
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(in_features=2048, out_features=12, bias=True)
# unfreeze layers
for param in model.parameters():
    param.requires_grad = True

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# summary(model = model,
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )

# Start the timer
start_time = timer()

# Setup training and save the results
train_models.train(model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=1,
                device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# save model
# FILE = "transfer_model.pth"
# torch.save(model.state_dict(), FILE)
