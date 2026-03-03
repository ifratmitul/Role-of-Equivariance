import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from e2cnn import gspaces, nn as enn
from model import FullyGEquivariantCNN10


def get_cifar10_loaders(batch_size=64):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2), \
           DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def train_model(model, trainloader, testloader, device, epochs=10, lr=0.001):
    print("Training model...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        correct, total_loss = 0, 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += outputs.argmax(1).eq(targets).sum().item()

        train_acc = 100. * correct / len(trainloader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.2f}% | Loss: {total_loss:.4f}")
        evaluate(model, testloader, device)


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            correct += outputs.argmax(1).eq(targets).sum().item()
    acc = 100. * correct / len(testloader.dataset)
    print(f"Test Accuracy: {acc:.2f}%\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullyGEquivariantCNN10()
    trainloader, testloader = get_cifar10_loaders(batch_size=64)

    # Train for 50 epochs
    train_model(model, trainloader, testloader, device, epochs=50)

    # Save the trained model
    torch.save(model.state_dict(), "g_equivariant_cnn10_cifar10.pth")
    print("Model saved to g_equivariant_cnn10_cifar10.pth")