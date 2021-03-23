import sys
import os
# print(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# from PIL import Image
# import numpy as np

from vision_transformer.vit import VisionTransformer

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat','deer','dog','frog','horse','ship','truck']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionTransformer(img_size=32, patch_size=4, in_chans=3, n_classes=10, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0.3, attn_p=0.3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the network

for epoch in range(100):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)

        # print('inputs shape: ', inputs.shape)
        # print('labels shape: ', labels.shape)
        # zero the parameter gradients
        optimizer.zero_grad()

        #
        # print(model(inputs).shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 0:
            print('[%d %5d] loss: %.3f' % (epoch +1, i+1, running_loss / 50))
            running_loss = 0.0

print('Finished Training')

# save model
PATH = './data/cifar_net.pth'
torch.save(model.state_dict(), PATH)

# Test settings
dataiter = iter(testloader)
images, labels = dataiter.next()

images = images.to(device)
labels = labels.to(device)

# test
net = VisionTransformer().to(device)
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)
print(predicted)
print(labels)

# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images).to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
##################################################################################