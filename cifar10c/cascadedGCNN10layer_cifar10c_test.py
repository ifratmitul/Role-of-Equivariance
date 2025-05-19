import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
from e2cnn.nn import R2Conv, GeometricTensor, FieldType, ReLU
from e2cnn.gspaces import Rot2dOnR2

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CascadedGCNN Model
class CascadedGCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):  # Adjusted for CIFAR-10
        super(CascadedGCNN, self).__init__()

        self.r2_act = Rot2dOnR2(4)  # Symmetry group (C4 rotations)
        self.input_type = FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])
        self.hidden_type = FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])

        # Standard Convolution
        self.conv1_standard = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Group Convolution
        self.conv1_group = R2Conv(self.hidden_type, self.hidden_type, kernel_size=3, padding=1)
        self.relu_group = ReLU(self.hidden_type)

        # Additional Convolutional Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Standard Convolution
        x_standard = self.conv1_standard(x)

        # Convert output of standard convolution to group tensor and apply group convolution
        x_group = GeometricTensor(x_standard, self.hidden_type)
        x_group = self.conv1_group(x_group)
        x_group = self.relu_group(x_group)

        # Convert back to a standard tensor
        x_group = x_group.tensor

        # Continue processing with additional layers
        x = self.pool(x_group)
        x = self.conv2(x)
        x = self.pool(self.conv3(x))
        x = self.conv4(x)
        x = self.pool(self.conv5(x))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        # Global Max Pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# FGSM Attack
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Clear gradients
    model.zero_grad()

    # Compute gradients
    loss.backward(retain_graph=True)  # Ensure the graph is retained for further steps if needed

    # Generate adversarial examples
    grad = images.grad.sign()
    perturbed_images = images + epsilon * grad

    # Detach the tensor to ensure no further graph is connected
    return perturbed_images.detach().clamp(0, 1)

# PGD Attack
def pgd_attack(model, images, labels, epsilon, alpha=0.01, num_steps=40):
    original_images = images.clone().detach().to(device)
    perturbed_images = images.clone().detach().to(device).requires_grad_(True)

    for _ in range(num_steps):
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        # Clear gradients
        model.zero_grad()

        # Compute gradients
        loss.backward(retain_graph=True)  # Retain the graph for iterative updates

        # Apply PGD update
        grad = perturbed_images.grad.sign()
        perturbed_images = perturbed_images + alpha * grad
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1).detach().requires_grad_(True)

    return perturbed_images.detach()

# CIFAR-10C dataset loader
def load_cifar10c_data(data_dir, batch_size):
    corruption_types = [f for f in os.listdir(data_dir) if f.endswith('.npy') and f != 'labels.npy']
    labels_path = os.path.join(data_dir, 'labels.npy')  # Load the labels separately

    # Load the labels
    labels = np.load(labels_path)
    labels = torch.tensor(labels, dtype=torch.long)

    data_loaders = {}

    for corruption in corruption_types:
        data_path = os.path.join(data_dir, corruption)
        data = np.load(data_path)  # Load the .npy file
        print(f"Loaded {corruption} with shape {data.shape}")

        # Normalize and permute data
        data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        dataset = torch.utils.data.TensorDataset(data, labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        data_loaders[corruption] = data_loader

    return data_loaders

# Load CIFAR-10C dataset
data_dir = './datasets/CIFAR-10-C'  # Path to CIFAR-10-C dataset
batch_size = 128
cifar10c_loaders = load_cifar10c_data(data_dir, batch_size)

# Load the saved model
model = CascadedGCNN().to(device)
state_dict = torch.load("cascadedGCNN10layer_cifar10.pth")

try:
    model.load_state_dict(state_dict)
    print("Model loaded successfully with all parameters.")
except RuntimeError as e:
    print(f"Error loading state_dict strictly: {e}")
    print("Attempting to load model state_dict non-strictly...")
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded non-strictly. Some layers may use default initialization.")

criterion = nn.CrossEntropyLoss()

# Evaluate model under attack
def evaluate_cifar10c(model, data_loaders, attack_fn, epsilon_values, combined_csv_file):
    model.eval()

    with open(combined_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Corruption', 'Attack', 'Epsilon', 'Accuracy'])

        for corruption, loader in data_loaders.items():
            print(f"\nEvaluating corruption type: {corruption}")
            for epsilon in epsilon_values:
                correct = 0
                total = 0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)

                    # Generate adversarial examples
                    adv_images = attack_fn(model, images, labels, epsilon)

                    # Evaluate model on adversarial examples
                    outputs = model(adv_images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                writer.writerow([corruption, attack_fn.__name__, epsilon, accuracy])
                print(f"Attack: {attack_fn.__name__}, Epsilon: {epsilon}, Accuracy: {accuracy:.2f}%")

# Epsilon values
epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]

# Combined results CSV file
combined_csv_file = 'cascadedGCNN10layer_cifar10c_adversarial_results.csv'

# Evaluate on CIFAR-10C and save combined results
print("\nEvaluating model on CIFAR-10C under FGSM attack...")
evaluate_cifar10c(model, cifar10c_loaders, fgsm_attack, epsilon_values, combined_csv_file)

print("\nEvaluating model on CIFAR-10C under PGD attack...")
evaluate_cifar10c(model, cifar10c_loaders, pgd_attack, epsilon_values, combined_csv_file)