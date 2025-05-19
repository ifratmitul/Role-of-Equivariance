import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from e2cnn.nn import R2Conv, GeometricTensor, FieldType, ReLU
from e2cnn.gspaces import Rot2dOnR2

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CascadedGCNN Model
class CascadedGCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=100):  # Changed num_classes to 100
        super(CascadedGCNN, self).__init__()

        self.r2_act = Rot2dOnR2(4)  # Symmetry group (C4 rotations)
        self.input_type = FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])
        self.hidden_type = FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])

        # Standard Convolution
        self.conv1_standard = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),  # Adjusted output channels
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Group Convolution
        self.conv1_group = R2Conv(self.hidden_type, self.hidden_type, kernel_size=3, padding=1)
        self.relu_group = ReLU(self.hidden_type)  # Equivariant ReLU

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
        x_group = self.relu_group(x_group)  # Use equivariant ReLU

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

# Preprocessing and dataset loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Model initialization, loss function, optimizer, and scheduler
model = CascadedGCNN().to(device)
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

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch + 1}/200], Loss: {running_loss / len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "cascadedGCNN10layer_cifar100.pth")
print("Training Complete. Model saved to 'cascadedGCNN10layer_cifar100.pth'.")

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

# Test the model
print("Testing on clean images...")
accuracy_clean = evaluate(model, test_loader)
print(f"Test Accuracy (Clean Images): {accuracy_clean:.2f}%")
