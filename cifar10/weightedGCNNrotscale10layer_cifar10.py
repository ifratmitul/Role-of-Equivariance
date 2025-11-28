# cifar10/weightedGCNNrotscale10layer_cifar10.py
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
model = ImprovedParallelGCNN(num_classes=10).to(device)  # Adjusted num_classes
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

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
torch.save(model.state_dict(), "weightedGCNNrotscale10layer_cifar10.pth")
print("Training Complete. Model saved to 'weightedGCNNrotscale10layer_cifar10.pth'.")

# Testing Function
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