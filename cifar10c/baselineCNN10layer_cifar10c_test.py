import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

# Define the CIFAR-10-C dataset class
class CIFAR10C(Dataset):
    def __init__(self, data_path, corruption_type, transform=None):
        self.data = np.load(os.path.join(data_path, f"{corruption_type}.npy"))
        self.labels = np.load(os.path.join(data_path, "labels.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations (same as for CIFAR-10)
transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the model class (ExtendedCNN)
class ExtendedCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(ExtendedCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExtendedCNN().to(device)
model.load_state_dict(torch.load('baselineCNN10layer_cifar10.pth'))
model.eval()
print("Model loaded from 'baselineCNN10layer_cifar10.pth'")

# FGSM Attack
def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    adversarial_images = images + perturbation
    adversarial_images = torch.clamp(adversarial_images, 0, 1)
    return adversarial_images

# PGD Attack
def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    original_images = images.clone().detach()
    for i in range(num_iter):
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        perturbation = alpha * images.grad.sign()
        images = images + perturbation
        images = torch.clamp(images, original_images - epsilon, original_images + epsilon)
        images = torch.clamp(images, 0, 1)
    return images

# Evaluate Model under Attack
def evaluate_attack(model, data_loader, attack_fn, epsilon, **kwargs):
    correct = 0
    total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        adversarial_images = attack_fn(model, images, labels, epsilon, **kwargs)
        outputs = model(adversarial_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Evaluate on CIFAR-10-C for multiple corruption types
data_path = './datasets/CIFAR-10-C'  # Path to CIFAR-10-C dataset
corruption_types = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'saturate', 'spatter'
]
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]

# Open CSV file for writing results
with open("baselineCNN_10layers_cifar10c_attack_results.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Corruption Type", "Epsilon", "FGSM Accuracy", "PGD Accuracy"])
    
    for corruption in corruption_types:
        print(f"\nEvaluating corruption: {corruption}")
        cifar10c_dataset = CIFAR10C(data_path, corruption, transform=transform)
        cifar10c_loader = DataLoader(cifar10c_dataset, batch_size=128, shuffle=False)
        
        for epsilon in epsilons:
            print(f"  Epsilon: {epsilon}")
            acc_fgsm = evaluate_attack(model, cifar10c_loader, fgsm_attack, epsilon)
            print(f"    FGSM Attack Accuracy: {acc_fgsm:.2f}%")
            acc_pgd = evaluate_attack(model, cifar10c_loader, pgd_attack, epsilon, alpha=0.01, num_iter=40)
            print(f"    PGD Attack Accuracy: {acc_pgd:.2f}%")
            
            # Write results to CSV
            csvwriter.writerow([corruption, epsilon, acc_fgsm, acc_pgd])

print("Results saved to baselineCNN_10layers_cifar10c_attack_results.csv")