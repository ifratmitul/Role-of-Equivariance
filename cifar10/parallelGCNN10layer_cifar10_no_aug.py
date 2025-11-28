#castar10/cascadedGCNN10layer_cifar10.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from e2cnn.nn import R2Conv, GeometricTensor, FieldType
from e2cnn.gspaces import Rot2dOnR2

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Improved ParallelGCNN Model
class ImprovedParallelGCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
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

        # Additional Convolutional Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=3, padding=1),
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
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Additional Convolutional Layer
        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_standard = self.conv1_standard(x)

        x_group = GeometricTensor(x, self.input_type)
        x_group = self.conv1_group(x_group)
        x_group = x_group.tensor

        x = torch.cat((x_standard, x_group), dim=1)

        x = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Basic Preprocessing (No Augmentation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize Model, Loss, Optimizer, and Scheduler
model = ImprovedParallelGCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# FGSM Attack
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward(retain_graph=True)
    grad = images.grad.sign()
    perturbed_images = images + epsilon * grad
    return torch.clamp(perturbed_images, 0, 1)

# PGD Attack
def pgd_attack(model, images, labels, epsilon, alpha=0.01, num_steps=40):
    original_images = images.clone().detach().to(device)
    perturbed_images = images.clone().detach().to(device).requires_grad_(True)

    for _ in range(num_steps):
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward(retain_graph=True)
        grad = perturbed_images.grad.sign()

        perturbed_images = perturbed_images + alpha * grad
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1).detach().requires_grad_(True)

    return perturbed_images

# Training Loop
print("Starting Training...")
for epoch in range(200):  # 200 epochs
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Combine clean and adversarial examples
        adversarial_images = fgsm_attack(model, images, labels, epsilon=0.01)
        combined_images = torch.cat((images, adversarial_images), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)

        # Forward pass
        outputs = model(combined_images)
        loss = criterion(outputs, combined_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += loss.item()
    
    scheduler.step()
    print(f"Epoch [{epoch + 1}/200], Loss: {running_loss / len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "parallelGCNN10layer_cifar10_no_aug.pth")
print("Training Complete. Model saved to 'parallelGCNN10layer_cifar10_no_aug.pth'.")

# Testing Functions
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def evaluate_adversarial(model, test_loader, attack_func, epsilon, **kwargs):
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Enable gradients explicitly for the input tensor
        images.requires_grad_()

        # Generate adversarial examples
        perturbed_images = attack_func(model, images, labels, epsilon, **kwargs)

        # Evaluate model on perturbed images
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total

# Test the model
print("Testing on clean images...")
accuracy_clean = evaluate(model, test_loader)
print(f"Test Accuracy (Clean Images): {accuracy_clean:.2f}%")

# Test with FGSM and PGD
epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
print("Testing on adversarial images...")
for epsilon in epsilon_values:
    # FGSM evaluation
    fgsm_accuracy = evaluate_adversarial(model, test_loader, fgsm_attack, epsilon)
    print(f"FGSM Accuracy (Epsilon = {epsilon}): {fgsm_accuracy:.2f}%")

    # PGD evaluation
    pgd_accuracy = evaluate_adversarial(model, test_loader, pgd_attack, epsilon, alpha=0.01, num_steps=40)
    print(f"PGD Accuracy (Epsilon = {epsilon}): {pgd_accuracy:.2f}%")
