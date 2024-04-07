'''Import required libraries'''
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler


'''Function for loading the dataset'''

train_path = "C:/Users/navee/OneDrive/Documents/dl/A2/inaturalist_12K/train"
test_path = "C:/Users/navee/OneDrive/Documents/dl/A2/inaturalist_12K/val"

def load_data(train_path, val_split=0.2, batch_size=150, data_augmentation=False):
    # Define transformations to apply to the images
    transform_list = [transforms.Resize((224, 224)),transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    if data_augmentation:
        transform_list.insert(0, transforms.RandomHorizontalFlip())
        transform_list.insert(0, transforms.RandomRotation(10))
    
    transform = transforms.Compose(transform_list)

    # Load dataset
    dataset = ImageFolder(train_path, transform=transform)
    num_classes = len(dataset.classes)
    
    # Create stratified train-validation split
    train_indices, val_indices = train_test_split(list(range(len(dataset))), 
                                                   test_size=val_split, 
                                                   stratify=dataset.targets)

    # Create DataLoader for training set
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    # Create DataLoader for validation set
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader, num_classes

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, filter_size, dense_units, activation_conv='relu', data_augmentation=False, batch_norm=False, dropout=0.0, filter_organization='same'):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(5):
            if filter_organization == 'same':
                current_num_filters = num_filters
            elif filter_organization == 'double':
                current_num_filters = num_filters * (2 ** i)
            elif filter_organization == 'halve':
                current_num_filters = num_filters // (2 ** i)

            self.conv_layers.append(nn.Conv2d(input_channels, current_num_filters, filter_size, padding=1))
            if batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(current_num_filters))
            if activation_conv == 'relu':
                self.conv_layers.append(nn.ReLU(inplace=True))
            elif activation_conv == 'gelu':
                self.conv_layers.append(nn.GELU())
            elif activation_conv == 'silu':
                self.conv_layers.append(nn.SiLU())
            elif activation_conv == 'mish':
                self.conv_layers.append(nn.Mish())
            #self.conv_layers.append(nn.Dropout2d(dropout))
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2))
            input_channels = current_num_filters

        input_size = 224  # Assuming input images are 224x224
        for _ in range(5):
            #input_size = (input_size // 2)
            input_size = (input_size - filter_size +3)
            input_size = (input_size - 2)//2 + 1
            
        self.fc = nn.Linear(input_channels * input_size * input_size, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
    
train_loader, val_loader, num_classes = load_data(train_path, val_split=0.2, batch_size=32, data_augmentation=True)
# Load pre-trained ResNet50 model
pretrained_resnet50 = models.resnet50(pretrained=True)

# Freeze all convolutional layers
for param in pretrained_resnet50.parameters():
    param.requires_grad = False

# Replace the classifier with a new fully connected layer

pretrained_resnet50.fc = nn.Linear(pretrained_resnet50.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_resnet50.fc.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_resnet50.to(device)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    # Training loop
    pretrained_resnet50.train()
    curr_loss = 0.0
    correct_train = 0
    total_train = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)  # Move tensors to GPU
        optimizer.zero_grad()
        outputs = pretrained_resnet50(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    epoch_loss = curr_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train
    # Validation loop (evaluate model performance on validation dataset)
    pretrained_resnet50.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)  # Move tensors to GPU
            outputs = pretrained_resnet50(imgs)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    validation_accuracy = correct_val / total_val
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, '
          f'Validation Accuracy: {validation_accuracy:.4f}')