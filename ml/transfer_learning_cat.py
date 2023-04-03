from timeit import default_timer as timer
import matplotlib.pyplot as plt
import torch
import torchvision
import random
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() /
                         len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# to utilise gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
batch_size_training = 48
batch_size_testing = 25


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
results = train(model=model,
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
FILE = "transfer_model.pth"
torch.save(model.state_dict(), FILE)
