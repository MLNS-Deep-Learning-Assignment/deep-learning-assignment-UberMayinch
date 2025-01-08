import torch
import torch.nn as nn
from torchvision import models

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#
test_data = []

# Load models
basic_model = torch.load('best_model.pth', map_location=device)
resnet_model = torch.load('best_resnet.pth', map_location=device)

print("\nEvaluating models on test data...")
basic_accuracy = evaluate_model(basic_model, test_data, device)
print(f"Basic CNN Accuracy: {basic_accuracy:.2f}%")

resnet_accuracy = evaluate_model(resnet_model, test_data, device)
print(f"ResNet Accuracy: {resnet_accuracy:.2f}%")