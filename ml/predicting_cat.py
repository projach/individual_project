import torch
import torchvision
import torch.nn.functional as F
import numpy
from typing import List, Tuple
from PIL import Image
from torch import nn
from torchvision import transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

CATEGORIES = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
              "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx"]


# 1. Take in a trained model, class names, image path, image size, a transform and target device
#add model number and whatever comes 
def transfer_learning_find_breed(img: Image,
                        model_num = 1,
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = device):

    #model
    if model_num == 1:
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CATEGORIES))
        model.load_state_dict(torch.load("D:/study_ml/ml/transfer_model.pth"))
    else:
        print("no model now")

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ###

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # 7. Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_image_pred_probs_softmax = F.softmax(target_image_pred_probs, dim=1)
    # 9. Convert prediction probabilities -> prediction labels
    # target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    # target_image_pred_label = torch.topk(target_image_pred_probs, 3, dim=1)
    top3_prob, top3_label = torch.topk(target_image_pred_probs_softmax, 3)

    #transform to numpy array
    top3_label_num = numpy.reshape(top3_label.cpu().numpy(),-1)
    top3_prob_num = numpy.reshape(top3_prob.cpu().numpy(),-1)
    
    #transform to named labels
    labels = []
    for i in top3_label_num:
        labels.append(CATEGORIES[i])
    
    return labels, top3_prob_num


# print(pred_and_plot_image(image_path="D:\study_ml\ml\Bombay_1.jpg"))
