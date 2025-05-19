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

class BaselineCNNWithParallelScaleAndRotEquivariance(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(BaselineCNNWithParallelScaleAndRotEquivariance, self).__init__()
        
        # Define rotational symmetry group
        self.rot_gspace = gspaces.Rot2dOnR2(N=8)  # Rotational symmetry with 8 discrete angles

        # Define field types for equivariant convolution
        self.feat_type_in_rot = enn.FieldType(self.rot_gspace, input_channels * [self.rot_gspace.trivial_repr])
        self.feat_type_out_rot = enn.FieldType(self.rot_gspace, 16 * [self.rot_gspace.regular_repr])

        # Standard convolutional layer
        self.conv_standard = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            nn.ReLU()
        )

        # Rotational-equivariant convolutional layer
        self.conv_equivariant_rot = enn.R2Conv(self.feat_type_in_rot, self.feat_type_out_rot, kernel_size=3, padding=1)
        self.relu_equivariant_rot = enn.ReLU(self.feat_type_out_rot)

        # Simulated scale-equivariant convolutions using separate scaled inputs
        self.conv_scale_0_5 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv_scale_1_0 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv_scale_2_0 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            nn.ReLU()
        )

        # Fusion layer to combine outputs of standard, rotational-equivariant, and scale-equivariant convolutions
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1),  # Adjust input channels to 192
            nn.BatchNorm2d(32),  # Add Batch Normalization
            nn.ReLU()
        )

        # Additional convolutional layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply standard convolution
        x_standard = self.conv_standard(x)

        # Apply rotational-equivariant convolution
        x_rot = enn.GeometricTensor(x, self.feat_type_in_rot)  # Wrap input for rotational-equivariant layer
        x_rot = self.conv_equivariant_rot(x_rot)
        x_rot = self.relu_equivariant_rot(x_rot)
        x_rot = x_rot.tensor  # Convert back to PyTorch tensor

        # Simulated scale-equivariant convolutions
        x_scale_0_5 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_scale_0_5 = self.conv_scale_0_5(x_scale_0_5)
        x_scale_0_5 = F.interpolate(x_scale_0_5, size=x.shape[2:], mode='bilinear', align_corners=False)

        x_scale_1_0 = self.conv_scale_1_0(x)

        x_scale_2_0 = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_scale_2_0 = self.conv_scale_2_0(x_scale_2_0)
        x_scale_2_0 = F.interpolate(x_scale_2_0, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate outputs from standard, rotational-equivariant, and scale-equivariant convolutions
        x_combined = torch.cat((x_standard, x_rot, x_scale_0_5, x_scale_1_0, x_scale_2_0), dim=1)

        # Apply fusion layer
        x = self.fusion(x_combined)

        # Pass through additional layers
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        # Global Max Pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])  # Pool to 1x1
        x = torch.flatten(x, 1)  # Flatten to feed into the FC layer

        # Fully connected layer
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
    model = BaselineCNNWithParallelScaleAndRotEquivariance().to(device)
    state_dict = torch.load("parallelGCNNrotscale10layer_cifar10.pth", map_location=device, weights_only=True)
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
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]

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
    results_df.to_csv("parallelGCNNrotscale10layer_cifar10c_test_results.csv", index=False)
    print("Results saved to parallelGCNNrotscale10layer_cifar10c_test_results.csv")

if __name__ == "__main__":
    main()