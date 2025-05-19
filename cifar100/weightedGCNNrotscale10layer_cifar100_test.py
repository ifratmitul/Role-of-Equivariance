import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from e2cnn.nn import R2Conv, GeometricTensor, FieldType
from e2cnn.gspaces import Rot2dOnR2

# Load the saved model
model_path = "weightedGCNNrotscale10layer_cifar100.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debugging utility
def debug_state_dict(state_dict, model):
    """Compare the keys in the state_dict and model parameters."""
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    print("\n[DEBUG] State Dict Check:")
    print(f"Missing Keys: {missing_keys}")
    print(f"Unexpected Keys: {unexpected_keys}")

# Improved ParallelGCNN Model with Learnable Weighted Fusion
class ImprovedParallelGCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=100):  
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

# Load the model and handle missing or unexpected keys
model = ImprovedParallelGCNN().to(device)

# Load the state dict with debugging
try:
    state_dict = torch.load(model_path, map_location=device)
    debug_state_dict(state_dict, model)  # Debug state_dict
    model.load_state_dict(state_dict, strict=False)
    print("[INFO] Model loaded successfully with potential missing keys.")
except Exception as e:
    print(f"[ERROR] Failed to load the model: {e}")

model.eval()

# Define FGSM Attack
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward(retain_graph=True)  # Retain the computation graph
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    return torch.clamp(perturbed_images, -1, 1)

# Define PGD Attack
def pgd_attack(model, images, labels, epsilon, alpha=0.01, iters=40):
    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True
    
    for _ in range(iters):
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        
        model.zero_grad()
        loss.backward(retain_graph=True)  
        data_grad = perturbed_images.grad.data
        
        # Update perturbed images
        perturbed_images = perturbed_images + alpha * data_grad.sign()
        
        # Clamp to ensure within valid pixel range
        perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, -1, 1)
        
        # Detach to clear the computation graph for the next iteration
        perturbed_images = perturbed_images.detach().clone()
        perturbed_images.requires_grad = True  # Re-enable gradients for the next iteration

    return perturbed_images

# Test Function
def test_with_attack(model, test_loader, attack, epsilon):
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        if attack == 'FGSM':
            adv_images = fgsm_attack(model, images, labels, epsilon)
        elif attack == 'PGD':
            adv_images = pgd_attack(model, images, labels, epsilon)
        
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epsilon: {epsilon}, Accuracy: {accuracy:.2f}%")
    return accuracy
    
# Load CIFAR-100 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = datasets.CIFAR100(root='./datasets', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Perform Adversarial Testing
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
attacks = ['FGSM', 'PGD']

print("Adversarial Testing...")
for attack in attacks:
    print(f"\nTesting with {attack} Attack:")
    for epsilon in epsilons:
        test_with_attack(model, test_loader, attack, epsilon)