import torch
import torchvision
import torch.nn.functional as F
import numpy
from typing import List, Tuple
from PIL import Image
from torch import nn
from torchvision import transforms

# to use gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# breeds that can be recognized by model
CATEGORIES = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
              "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx"]

# states of models path
TransferLearningPath = "D:/individual_project/ml/transfer_model.pth"
CNNLearningPath = "D:/individual_project/ml/cnn_model.pth"

# add the image the model number the image size is added automaticaly a transformer
# is added into the code and the device to go into gpu


def transfer_learning_find_breed(img: Image,
                                 model_str: str,
                                 image_size: Tuple[int, int] = (224, 224),
                                 transform: torchvision.transforms = None,
                                 device: torch.device = device):

    # models used
    if model_str == "Transfer Learning(resnet50)":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CATEGORIES))
        model.load_state_dict(torch.load(
            TransferLearningPath))

    elif model_str == "My Model":
        class ConvNet(nn.Module):
            def __init__(self):
                super(ConvNet, self).__init__()
                self.conv1 = nn.Conv2d(
                    3, 16, kernel_size=3, stride=1, padding=1)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(
                    16, 32, kernel_size=3, stride=1, padding=1)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv3 = nn.Conv2d(
                    32, 64, kernel_size=3, stride=1, padding=1)
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
        model = ConvNet()
        model.fc = nn.Linear(model.fc.in_features, len(CATEGORIES))
        model.load_state_dict(torch.load(
            CNNLearningPath))

    # transform the image for the correct image input into the model
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    ### Predict image ###

    # send model to gpu
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # get probabilities from 0-1
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_probs_softmax = F.softmax(target_image_pred_probs, dim=1)

    # Convert prediction probabilities -> prediction labels and get the top 3 labels
    top3_prob, top3_label = torch.topk(target_image_pred_probs_softmax, 3)

    # transform to numpy array (we need to return to cpu to do that)
    top3_label_num = numpy.reshape(top3_label.cpu().numpy(), -1)
    top3_prob_num = numpy.reshape(top3_prob.cpu().numpy(), -1)

    # transform to named labels
    labels = []
    for i in top3_label_num:
        labels.append(CATEGORIES[i])

    return labels, top3_prob_num
