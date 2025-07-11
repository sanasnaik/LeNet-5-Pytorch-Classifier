from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mat_dataset import *

def get_mnist_loaders(batch_size=1):
    # Define transform: pad 28x28 to 32x32 and convert to tensor
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])

    # Load MNIST train and test datasets
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

"""Load Dataset"""


"""Load Dataset 2"""
def get_mnist_loaders_2(batch_size=1):
    # Define transform: pad 28x28 to 32x32 and convert to tensor
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])

    # Load MNIST train and test datasets
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Load MAT dataset
    root_dir = os.path.join(os.path.dirname(__file__), "training_batches")
    mat_dataset = MATDataset(mat_dir=root_dir)  # Adjust path as needed
    combined_dataset = torch.utils.data.ConcatDataset([trainset, mat_dataset])

    # Create DataLoaders
    trainloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader