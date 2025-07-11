# Command Instructions:
# To run this script and evaluate LeNet1 on the original MNIST test set,
# execute the following command in your terminal:
#
#     python test1.py
#
# Make sure the file LeNet1.pth and the image folder is present in the same directory.


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data import *
import torch
import numpy as np
import torchvision
from le_net_5 import *
import torch.serialization
from torch.serialization import safe_globals
from rbf_classifier import *
from margin_loss import *
from bitmap_generation import *
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from torchvision.utils import save_image


 
def test(dataloader,model):

    """Test Code"""
    model.eval()
    test_total = 0
    test_correct = 0

    all_preds = []
    all_labels = []
    most_confusing = {}

    with torch.no_grad():  # No gradient computation during evaluation
        for images, labels in dataloader:

            outputs = model(images)
            _, predicted = outputs.min(dim=1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(images.size(0)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()

                if true_label != pred_label:
                    confidence = outputs[i, pred_label].item()
                    if (true_label not in most_confusing or
                            confidence < most_confusing[true_label][2]):
                        most_confusing[true_label] = (images[i], pred_label, confidence)
                    

    test_error_rate = 1 - (test_correct / test_total)
    #test_errors.append(test_error_rate)
    print(f"Test Error Rate: {test_error_rate:.4f}")                                                                                                                                                                          

    test_accuracy= 1 - test_error_rate

    print("test accuracy:", test_accuracy)

    with open("model_one_metrics.json", "r") as f:
        metrics = json.load(f)

    metrics["final_test_error"] = test_error_rate

    with open("model_one_metrics.json", "w") as f:
        json.dump(metrics, f)


    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    with open("model_one_metrics.json") as f:
        data = json.load(f)

    train_errors = data["train_errors"]
    final_test_error = data["final_test_error"]

    plt.plot([e * 100 for e in train_errors], label="Train Error", marker='o')
    plt.axhline(test_error_rate * 100, color='red', linestyle='--', label="Final Test Error")

    plt.xlabel("Epoch")
    plt.ylabel("Error Rate (%)")
    plt.title("Train vs Final Test Error")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save most confusing examples
    os.makedirs("most_confusing_digits", exist_ok=True)
    for true_digit in most_confusing:
        image_tensor, predicted_digit, _ = most_confusing[true_digit]
        save_path = f"most_confusing_digits/true_{true_digit}_pred_{predicted_digit}.png"
        save_image(image_tensor, save_path)
        print(f"Saved most confusing digit {true_digit} (misclassified as {predicted_digit}) to {save_path}")
        

 

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    trainloader, testloader = get_mnist_loaders(batch_size=1)

    # Recreate model structure first
    directory = os.path.join(os.path.dirname(__file__), "digits updated")
    prototypes = generate_bitmap_prototypes(directory)
    rbf_layer = RBFClassifier(prototypes=prototypes)
    model = LeNet5(rbf_layer=rbf_layer)

    # Then load the weights
    model.load_state_dict(torch.load("LeNet1.pth"))
    model.eval()
    test(testloader,model)

    

 

if __name__=="__main__":

    main()
