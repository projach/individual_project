import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#device configuration to use gpu if there is cuda available 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
input_size = 784 #because the photos are 28*28
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))])



#import mnist data
train_dataset = torchvision.datasets.MNIST(root='./data', train= True, 
    transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor())

#data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, lables = next(examples)

print(samples.shape,lables.shape)

##### this is used to show 6 numbers from the dataset(you can only show <= 6)
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()

#the actual neural network
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

model = NeuralNet(input_size,hidden_size,num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i,(images,lables) in enumerate(train_loader):
        #100, 1, 28, 28
        #100, 784
        images = images.reshape(-1, 28*28).to(device)
        lables = lables.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, lables)
        #backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"epoch{epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss={loss.item():.4f}")

FILE = "model.pth"
torch.save(model.state_dict(), FILE)

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, lables in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        lables = lables.to(device)
        outputs = model(images)
        #value, index
        _, predictions = torch.max(outputs,1)
        n_samples += lables.shape[0]
        n_correct += (predictions == lables).sum().item()
    acc = 100*n_correct/n_samples
    print(f'accuracy = {acc}')
