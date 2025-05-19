import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from e2cnn.nn import R2Conv, GeometricTensor, FieldType
from e2cnn.gspaces import Rot2dOnR2
from torch.utils.data import DataLoader, TensorDataset
from e2cnn import gspaces, nn as enn
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="e2cnn")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Improved ParallelGCNN Model with Learnable Weighted Fusion
class ImprovedParallelGCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):  # Adjusted num_classes for CIFAR-10
        super(ImprovedParallelGCNN, self).__init__()

        self.r2_act = Rot2dOnR2(4)  # Symmetry group
        self.input_type = FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])
        self.hidden_type = FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])

        # Standard and Group Convolutions
        self.conv1_standard = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv1_group = R2Conv(self.input_type, self.hidden_type, kernel_size=3, padding=1)

        # Bottleneck Layer to match channels
        self.bottleneck = nn.Conv2d(64, 16, kernel_size=1)

        # Learnable weights for weighted sum
        self.weight_standard = nn.Parameter(torch.tensor(0.5))  # Initialized to 0.5
        self.weight_group = nn.Parameter(torch.tensor(0.5))  # Initialized to 0.5

        # Additional Convolutional Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply Standard Convolution
        x_standard = self.conv1_standard(x)

        # Apply Group Convolution
        x_group = GeometricTensor(x, self.input_type)
        x_group = self.conv1_group(x_group)
        x_group = x_group.tensor

        # Apply Bottleneck to match channels
        x_group = self.bottleneck(x_group)

        # Normalize weights using softmax for numerical stability
        w_standard = F.softmax(torch.stack([self.weight_standard, self.weight_group]), dim=0)[0]
        w_group = F.softmax(torch.stack([self.weight_standard, self.weight_group]), dim=0)[1]

        # Combine outputs as weighted sum
        x_combined = w_standard * x_standard + w_group * x_group

        # Apply ReLU and Pooling
        x = self.pool(F.relu(x_combined))

        # Pass through additional convolutional layers
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        # Global Max Pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# FGSM attack
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward(retain_graph=True)
    grad = images.grad.sign()
    perturbed_images = images + epsilon * grad
    return torch.clamp(perturbed_images, 0, 1)

# PGD attack
def pgd_attack(model, images, labels, epsilon, alpha=0.01, num_steps=40):
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()

    for _ in range(num_steps):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward(retain_graph=True)  # Fix: Retain graph for multiple backward passes
        grad = perturbed_images.grad.sign()
        perturbed_images = perturbed_images.detach() + alpha * grad
        perturbed_images = torch.max(torch.min(perturbed_images, original_images + epsilon), original_images - epsilon)
        perturbed_images = perturbed_images.clamp(0, 1)

    return perturbed_images

# Load CIFAR-10-C dataset
def load_cifar10c(data_dir, corruption_type):
    file_path = os.path.join(data_dir, f"{corruption_type}.npy")
    data = np.load(file_path)
    return data

# Evaluation on clean and adversarial images
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def evaluate_adversarial(model, dataloader, attack_func, epsilon, **kwargs):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        perturbed_images = attack_func(model, images, labels, epsilon, **kwargs)
        
        # Debug statements
        #print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
        #print(f"Perturbed images shape: {perturbed_images.shape}")
        
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Main function
def main():
    model = ImprovedParallelGCNN().to(device)
    state_dict = torch.load("weightedGCNNrotscale10layer_cifar10.pth", map_location=device, weights_only=True)
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=False)

    corruption_types = [
        "gaussian_noise", "shot_noise", "impulse_noise", 
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", 
        "snow", "frost", "fog", "brightness", 
        "contrast", "elastic_transform", "pixelate", "jpeg_compression", 
        "speckle_noise", "gaussian_blur", "saturate", "spatter"
    ]
    data_dir = "./datasets/CIFAR-10-C"  # Path to CIFAR-10-C dataset
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

    results = []

    for corruption in corruption_types:
        print(f"\nEvaluating on corruption: {corruption}")
        data = load_cifar10c(data_dir, corruption)
        labels = np.load(os.path.join(data_dir, "labels.npy"))
        dataset = TensorDataset(torch.tensor(data).permute(0, 3, 1, 2).float() / 255, torch.tensor(labels))
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        clean_accuracy = evaluate(model, dataloader)
        results.append({"Corruption": corruption, "Epsilon": 0.0, "Attack": "Clean", "Accuracy": clean_accuracy})
        print(f"Clean Accuracy: {clean_accuracy:.2f}%")

        for epsilon in epsilon_values:
            fgsm_accuracy = evaluate_adversarial(model, dataloader, fgsm_attack, epsilon)
            results.append({"Corruption": corruption, "Epsilon": epsilon, "Attack": "FGSM", "Accuracy": fgsm_accuracy})
            print(f"FGSM Accuracy (Epsilon = {epsilon}): {fgsm_accuracy:.2f}%")

            pgd_accuracy = evaluate_adversarial(model, dataloader, pgd_attack, epsilon, alpha=0.01, num_steps=40)
            results.append({"Corruption": corruption, "Epsilon": epsilon, "Attack": "PGD", "Accuracy": pgd_accuracy})
            print(f"PGD Accuracy (Epsilon = {epsilon}): {pgd_accuracy:.2f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv("weightedGCNNrotscale10layer_cifar10c_test_results.csv", index=False)
    print("Results saved to weightedGCNNrotscale10layer_cifar10c_test_results.csv")

if __name__ == "__main__":
    main()