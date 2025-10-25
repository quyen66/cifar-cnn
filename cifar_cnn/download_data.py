from torchvision import datasets, transforms
import os

os.makedirs('./data', exist_ok=True)
transform = transforms.ToTensor()

print("Downloading CIFAR-10...")
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
print(f"✅ Success! Train: {len(trainset)}, Test: {len(testset)}")
