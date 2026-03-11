import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from e2cnn import gspaces, nn as enn
from model import FullyGEquivariantCNN10

CLIP_MIN = -2.43
CLIP_MAX = 2.75


def fgsm_attack(model, images, labels, epsilon, clip_min=CLIP_MIN, clip_max=CLIP_MAX):
    adv_images = images.clone().detach().to(images.device)
    adv_images.requires_grad_(True)

    outputs = model(adv_images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = adv_images.grad.data
    perturbed_image = adv_images + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, clip_min, clip_max)

    return perturbed_image.detach()


def pgd_attack(model, images, labels, epsilon, alpha=0.01, num_iter=40,
               clip_min=CLIP_MIN, clip_max=CLIP_MAX, random_start=True):
    ori_images = images.clone().detach()
    adv_images = images.clone().detach()

    # Random initialization within epsilon-ball (required for true PGD)
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, clip_min, clip_max).detach()

    for _ in range(num_iter):
        adv_images_var = adv_images.clone().detach().requires_grad_(True)

        outputs = model(adv_images_var)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = adv_images_var.grad.data
        adv_images = adv_images + alpha * data_grad.sign()

        # Project back to epsilon-ball, then to valid input range
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + eta, clip_min, clip_max).detach()

    return adv_images


def evaluate_clean_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Clean Accuracy: {accuracy:.2f}%")
    return accuracy


def evaluate_attack(model, loader, attack_fn, epsilon, attack_name, device):
    # Use torch.enable_grad() only where gradients are needed.
    model.eval()
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples with gradients enabled, model stays in eval mode
        with torch.enable_grad():
            adversarial_images = attack_fn(model, images, labels, epsilon)

        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(adversarial_images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        model.zero_grad()

        if batch_idx % 20 == 0:
            current_acc = 100 * correct / total if total > 0 else 0
            print(f"  Batch {batch_idx}, Current {attack_name} Acc: {current_acc:.2f}%")

    accuracy = 100 * correct / total
    print(f"{attack_name} Attack (Epsilon: {epsilon}) - Final Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FullyGEquivariantCNN10(input_channels=3, num_classes=10).to(device)

    try:
        model.load_state_dict(torch.load("g_equivariant_cnn10_cifar10.pth", map_location=device))
        print("Model loaded successfully from 'g_equivariant_cnn10_cifar10.pth'")
    except FileNotFoundError:
        print("Error: Model file 'g_equivariant_cnn10_cifar10.pth' not found!")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("\n" + "="*50)
    print("ADVERSARIAL ROBUSTNESS EVALUATION")
    print("="*50)

    clean_acc = evaluate_clean_accuracy(model, test_loader, device)

    print("\nAdversarial Attack Results:")
    print("-" * 30)

    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    results = {
        'clean_accuracy': clean_acc,
        'fgsm': {},
        'pgd': {}
    }

    for epsilon in epsilons:
        print(f"\nEpsilon: {epsilon}")
        fgsm_acc = evaluate_attack(model, test_loader, fgsm_attack, epsilon, "FGSM", device)
        pgd_acc  = evaluate_attack(model, test_loader, pgd_attack,  epsilon, "PGD",  device)
        results['fgsm'][epsilon] = fgsm_acc
        results['pgd'][epsilon]  = pgd_acc

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Clean Accuracy: {results['clean_accuracy']:.2f}%")
    print("\nFGSM Attack Results:")
    for eps, acc in results['fgsm'].items():
        print(f"  Epsilon {eps}: {acc:.2f}%")
    print("\nPGD Attack Results:")
    for eps, acc in results['pgd'].items():
        print(f"  Epsilon {eps}: {acc:.2f}%")

    results_file = "adversarial_results_fullgcnn_cifar10.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to '{results_file}'")

    summary_file = "adversarial_summary_fullgcnn_cifar10.txt"
    with open(summary_file, 'w') as f:
        f.write("Adversarial Robustness Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write("Model: FullyGEquivariantCNN10\n")
        f.write("Dataset: CIFAR-10\n")
        f.write(f"Clean Accuracy: {results['clean_accuracy']:.2f}%\n\n")
        f.write("FGSM Attack Results:\n")
        for eps, acc in results['fgsm'].items():
            f.write(f"  Epsilon {eps}: {acc:.2f}%\n")
        f.write("\nPGD Attack Results:\n")
        for eps, acc in results['pgd'].items():
            f.write(f"  Epsilon {eps}: {acc:.2f}%\n")
    print(f"Summary saved to '{summary_file}'")
    print("\n" + "="*50)