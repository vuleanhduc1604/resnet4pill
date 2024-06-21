import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet101
import torch.nn as nn 
import os
from torch.utils.data import Dataset
from PIL import Image

class ImageListDataset(Dataset):
    def __init__(self, img_list, img_dir, transform=None):
        self.img_list = img_list
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name
    
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image to 512x512
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def infer_with_model(model, dataloader):
    model.to(device)
    model.eval()
    all_outputs = []
    all_img_names = []

    with torch.no_grad():
        for inputs, img_names in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_img_names.extend(img_names)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    
    model.to('cpu')
    
    return all_outputs, all_img_names

def ensemble_predict(models, dataloader):
    all_model_outputs = []

    # Infer with each model and collect all predictions
    for model in models:
        model_outputs, all_img_names = infer_with_model(model, dataloader)
        all_model_outputs.append(model_outputs)

    # Stack all model outputs and average them
    avg_outputs = torch.mean(torch.stack(all_model_outputs), dim=0)

    return avg_outputs, all_img_names

def save_predictions_to_file(predictions, img_names, output_file):
    _, predicted_classes = predictions.max(dim=1)
    with open(output_file, 'w') as f:
        for img_name, predicted_class in zip(img_names, predicted_classes):
            f.write(f"{img_name} {predicted_class.item()}\n")

# Example usage
img_dir = 'private/images'

num_imgs = len(os.listdir(img_dir))

img_list = [str(x) + '.jpg' for x in range(num_imgs) ]  # Your list of image file names
dataset = ImageListDataset(img_list, img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

root_dir = '512_5_14' 
models = []
for i in range(15): 
    try:
        model = resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for 100 classes
        model.load_state_dict(torch.load( root_dir + f'/resnet101_model_{i}.pth'))
        models.append(model)
    except: 
        continue

# Assuming 'models' is a list of your trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_predictions, all_img_names = ensemble_predict(models, dataloader)
save_predictions_to_file(all_predictions, all_img_names, 'predictions.txt')
