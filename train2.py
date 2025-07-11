#import statements 
from bitmap_generation import *
from data import *
from le_net_5_updated import *
from rbf_classifier import *
from margin_loss import *
import torch
import torch.optim as optim
import json
import os


import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys




"""Load Bitmaps"""
root_dir = os.path.join(os.path.dirname(__file__), "digits updated")
prototypes = generate_bitmap_prototypes(root_dir)
print("Bitmap prototypes generated:", prototypes.shape)  # [10, 84]

"""Load Dataset"""
trainloader, testloader = get_mnist_loaders_2(batch_size=1)

"""Initialization"""
rbf_layer = RBFClassifier(prototypes=prototypes)
model = LeNet5(rbf_layer=rbf_layer)
criterion = LeNetRBFMarginLoss(j=0.1)  # Margin loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



"""Training Loop"""
train_errors = []
test_errors = []

num_epochs = 20

for epoch in range(num_epochs):

    model.train()
    total, correct, running_loss = 0, 0, 0.0

    for images, labels in trainloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.min(outputs, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    error_rate = 1 - (correct/total)
    train_errors.append(error_rate)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Error Rate: {error_rate:.4f}')


# Save final model
torch.save(model.state_dict(), "LeNet2.pth")
print("Final model saved to LeNet2.pth")

# Save training errors
with open("model_two_metrics.json", "w") as f:
    json.dump({"train_errors": train_errors}, f)

    

