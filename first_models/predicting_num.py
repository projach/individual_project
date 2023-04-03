import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image



#load model
device = torch.device('cuda:0')
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


input_size = 784 #because the photos are 28*28
hidden_size = 100
num_classes = 10
model = NeuralNet(input_size,hidden_size,num_classes)

PATH = 'D:\study_ml\ml\model.pth'
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()


#transform image to torch
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.Normalize((0.1307),(0.3081))]) 
    return transform(image_bytes).unsqueeze(0)

#do the prediction
def get_prediction(image_tensor):
        image = image_tensor.reshape(-1, 28*28).to(device)
        outputs = model(image)
        probs = nn.functional.softmax(outputs, dim=1)
        #value, index
        _, predicted = torch.topk(probs,3)
        return predicted

# with open("number4.png", "rb") as f:
#     image_b = f.read()
#     print(get_prediction(transform_image(image_b)))