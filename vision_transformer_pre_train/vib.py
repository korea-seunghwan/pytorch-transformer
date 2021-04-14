import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vision_transformer_pre_train.vision_transformer import VisionTransformer
from vision_transformer_pre_train.losses import CustomContrastiveLoss

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='vision_transformer_pre_train/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='vision_transformer_pre_train/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#####################################################
## image size check ##
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# print(images.shape)
#####################################################

net = VisionTransformer(img_size=32, patch_size=4, in_chans=3, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4, qkv_bias=True)

# criterion = nn.KLDivLoss(reduction='sum')
criterion = CustomContrastiveLoss(margin=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        mask_outputs, original_output = net(inputs)
        # print("mask_outputs shape: ", mask_outputs)
        # print("original_output shape: ", original_output)

        loss = criterion(original_output, mask_outputs, torch.tensor(1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")