import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from p27_model import Tudui

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.247, 0.243, 0.261])
])

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_paths = {"39": "D:/作业/project1/logg/log_train_seed39/best.pth",
               "42": "D:/作业/project1/logg/log_train_seed42/best.pth",
               "45": "D:/作业/project1/logg/log_train_seed45/best.pth"}

criterion = nn.CrossEntropyLoss()

for seed_name, path in model_paths.items():
    model = Tudui().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    print(f"avg_loss = {avg_loss:.4f}")
    print(f"top1 = {acc * 100:.2f}%")
    print(f"seed = {seed_name}\n")


