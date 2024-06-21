from torchvision import transforms
import dataset # Custom dataset
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet101
import torch.optim as optim
import torch.nn as nn

from torchvision.transforms import functional as F

num_models = 2
import os 
root_dir = 'crop_512'
# make dir if not exist
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

batch_size = 8
log_file = 'log.txt'
log_file = os.path.join(root_dir, log_file)

class Crop10Percent(object):
    def __call__(self, img):
        width, height = img.size
        crop_size = (int(height * 0.8), int(width * 0.8))
        top = (height - crop_size[0]) // 2
        left = (width - crop_size[1]) // 2
        
        # print(top, left, crop_size[0], crop_size[1])
        return F.crop(img, top, left, crop_size[0], crop_size[1])

    def __repr__(self):
        return self.__class__.__name__ + '()'

transform = transforms.Compose([
    Crop10Percent(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensemble_predict(models, dataloader):
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = [model(inputs) for model in models]
            avg_outputs = torch.mean(torch.stack(outputs), dim=0)
            all_outputs.append(avg_outputs)
            all_labels.append(labels)

    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_outputs, all_labels

def calculate_accuracy(predictions, labels):
    _, predicted_classes = predictions.max(dim=1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def test(models, test_dataset ):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_predictions, all_labels = ensemble_predict(models, test_loader)
    test_accuracy = calculate_accuracy(all_predictions, all_labels)
    print(f"    Overall Test Accuracy: {test_accuracy * 100:.2f}%")
    with open(log_file, 'a') as f:
        f.write(f"===================  Overall Test Accuracy: {test_accuracy * 100:.2f}% \n")
    return test_accuracy

def train_model(model, dataloader, criterion, optimizer, num_epochs=30, model_fold=0):
    model.train()
    with open(log_file, 'a') as f:
                f.write(f'\n------------------------------------------------------------------------------------\n')
    test_dataset = dataset.CustomImageDataset(annotations_file= f'training/fold{model_fold}_test.txt', img_dir='training/images', transform=transform)
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # log to txt file 
            with open(log_file, 'a') as f:
                f.write(f'Model_fold {model_fold}, Epoch {epoch}, Loss: {loss.item()}\n')
        
        val_acc = test([model], test_dataset)
            
            
            
# Parameters

models = []

for i in range(num_models):
    model = resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for 100 classes
    # model.load_state_dict(torch.load('resnet101_model_0.pth'))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Get a unique DataLoader for each model
    
    train_dataset = dataset.CustomImageDataset(annotations_file=f'training/fold{i}_train.txt', img_dir='training/images', transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Training model {i}") 
    train_model(model, dataloader, criterion, optimizer, num_epochs=40, model_fold=i)

    # Save model to disk
    torch.save(model.state_dict(), root_dir + f'/resnet101_model_{i}.pth')
    models.append(model)

