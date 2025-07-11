##this is a reference code##
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision


def test(dataloader,model):
    sum_=0
    for batch, (image, label) in enumerate(dataloader):
        target =torch.nn.functional.one_hot(label,num_classes=10).float()
        sum_+=((torch.argmin(model(image),dim=1)==torch.argmax(target,dim=1)).sum().item())
        test_accuracy=sum_/(batch+1)
        print(test_accuracy)
    print("test accuracy:", test_accuracy)



def main():
    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')
    mnist_test=mnist.MNIST(split="test",transform=pad)
    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)
    model = torch.load("LeNet.pth")
    test(test_dataloader,model)
    
if __name__=="__main__":
    main()
