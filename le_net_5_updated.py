#import statements 
import torch
import torch.nn as nn
import torch.nn.functional as F


"""LeNet5 Class Structure"""

class LeNet5(nn.Module):
    def __init__(self, rbf_layer):
        super(LeNet5, self).__init__()
        # C1: Convolutional layer; input: 32x32 → output: 28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # S2: Subsampling (Average Pooling); output: 14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C3: Conv (14x14 → 10x10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # S4: Pool (10x10 → 5x5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C5: Conv (5x5 -> 1x1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # F6: Fully Connected Layer (120 -> 84)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.rbf = rbf_layer

        
        self.dropout_fc = nn.Dropout(p=0.5)


    # def scaled_tanh(self, x):
    #   return 1.7159 * F.tanh((2/3) * x)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # Apply C1 + activation
        x = self.pool1(x)             # Apply S2
           
                          
        x = F.relu(self.conv2(x))     # C3
        x = self.pool2(x)             # S4
       
                            
        x = F.relu(self.conv3(x))     # C5
        x = x.view(-1, 120)                     # Flatten

        x = F.relu(self.fc1(x))       # F6
        x = self.dropout_fc(x)        # Dropout
        x = self.rbf(x)                         # RBF layer

        return x