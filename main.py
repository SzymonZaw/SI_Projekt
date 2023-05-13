# 0. Biblioteki
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# 1. Definiowanie modelu sieci neuronowej
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # warstwy konwolucyjne
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # warstwy w pełni połączone
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # przepływ danych przez sieć neuronową
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 2. Przygotowanie danych
#Transformacje dla danych treningowych i testowych
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#Ładowanie danych treningowych i testowych
train_data = datasets.ImageFolder('train', transform=train_transforms)
test_data = datasets.ImageFolder('test', transform=test_transforms)

# 3. Tworzenie obiektów DataLoader
#Dataloader - obiekt umożliwiający ładowanie danych do modelu w porcjach, z ustawieniem losowego przemieszania danych
trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)

# 4. Trenowanie i testowanie sieci neuronowej
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(50):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # Ustawienie modelu na trenowanie
        net.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 7 == 6:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0
    #Ustawienie modelu na testowanie
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #Wyświetlenie próbek testowych
            #fig, ax = plt.subplots(figsize=(5, 5))
            #ax.imshow(images[0].permute(1, 2, 0))
            #ax.set_title(f'Predicted: {predicted[0]}, Actual: {labels[0]}')
            #ax.axis('off')
            #plt.show()
    print('Testing accuracy: %.3f %%' % (100 * correct / total))
