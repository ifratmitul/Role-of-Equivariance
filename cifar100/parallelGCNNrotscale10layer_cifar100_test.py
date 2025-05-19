import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from e2cnn import gspaces, nn as enn

class BaselineCNNWithParallelScaleAndRotEquivariance(nn.Module):
    def __init__(self, input_channels=3, num_classes=100):  
        super(BaselineCNNWithParallelScaleAndRotEquivariance, self).__init__()
        
        # Define rotational symmetry group
        self.rot_gspace = gspaces.Rot2dOnR2(N=8)  # Rotational symmetry with 8 discrete angles

        # Define field types for equivariant convolution
        self.feat_type_in_rot = enn.FieldType(self.rot_gspace, input_channels * [self.rot_gspace.trivial_repr])
        self.feat_type_out_rot = enn.FieldType(self.rot_gspace, 16 * [self.rot_gspace.regular_repr])

        # Standard convolutional layer
        self.conv_standard = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Rotational-equivariant convolutional layer
        self.conv_equivariant_rot = enn.R2Conv(self.feat_type_in_rot, self.feat_type_out_rot, kernel_size=3, padding=1)
        self.relu_equivariant_rot = enn.ReLU(self.feat_type_out_rot)

        # Simulated scale-equivariant convolutions
        self.conv_scale_0_5 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_scale_1_0 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_scale_2_0 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Additional convolutional layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
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
        x_standard = self.conv_standard(x)

        x_rot = enn.GeometricTensor(x, self.feat_type_in_rot)
        x_rot = self.conv_equivariant_rot(x_rot)
        x_rot = self.relu_equivariant_rot(x_rot)
        x_rot = x_rot.tensor

        x_scale_0_5 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_scale_0_5 = self.conv_scale_0_5(x_scale_0_5)
        x_scale_0_5 = F.interpolate(x_scale_0_5, size=x.shape[2:], mode='bilinear', align_corners=False)

        x_scale_1_0 = self.conv_scale_1_0(x)

        x_scale_2_0 = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_scale_2_0 = self.conv_scale_2_0(x_scale_2_0)
        x_scale_2_0 = F.interpolate(x_scale_2_0, size=x.shape[2:], mode='bilinear', align_corners=False)

        x_combined = torch.cat((x_standard, x_rot, x_scale_0_5, x_scale_1_0, x_scale_2_0), dim=1)

        x = self.fusion(x_combined)

        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNNWithParallelScaleAndRotEquivariance().to(device)

try:
    model.load_state_dict(torch.load("parallelGCNNrotscale10layer_cifar100.pth", map_location=device), strict=False)
    print("Model loaded with partial state_dict.")
except RuntimeError as e:
    print("Error loading model:", e)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR100(root='./datasets', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    adversarial_images = images + perturbation
    adversarial_images = torch.clamp(adversarial_images, -1, 1)
    return adversarial_images

def pgd_attack(model, images, labels, epsilon, alpha=0.01, num_iter=40):
    adv_images = images.clone().detach().to(device)
    adv_images.requires_grad = True
    for _ in range(num_iter):
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        perturbation = alpha * adv_images.grad.sign()
        adv_images = adv_images + perturbation
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, -1, 1).detach_()
        adv_images.requires_grad = True
    return adv_images

def evaluate_attack(model, loader, attack_fn, epsilon, attack_name):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adversarial_images = attack_fn(model, images, labels, epsilon)
        outputs = model(adversarial_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"{attack_name} Attack (Epsilon: {epsilon}) - Accuracy: {100 * correct / total:.2f}%")

epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]

for epsilon in epsilons:
    evaluate_attack(model, test_loader, fgsm_attack, epsilon, "FGSM")
    evaluate_attack(model, test_loader, pgd_attack, epsilon, "PGD")