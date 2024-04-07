import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import argparse

def load_data(train_path, val_split=0.2, batch_size=150, data_augmentation=False):
    transform_list = [transforms.Resize((224, 224)),transforms.ToTensor()]
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
    
def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average loss and accuracy
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    
    return avg_val_loss, val_accuracy

def train_model(config):
    wandb.login()
    wandb.init(entity=config.wandb_entity, project=config.wandb_project)
    
    # Load data
    train_loader, val_loader, num_classes = load_data(config.train_path, val_split=0.2, batch_size=config.batch_size, data_augmentation=config.data_augmentation)

    # Define model
    model = CNN(input_channels=3, num_classes=num_classes, num_filters=config.num_filters,
                filter_size=config.filter_size, dense_units=config.dense_units, activation_conv=config.activation_conv,
                data_augmentation=config.data_augmentation, batch_norm=config.batch_norm, dropout=config.dropout,
                filter_organization=config.filter_organization)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Calculate training loss and accuracy
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train

        # Validation
        avg_val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Log metrics to wandb
        wandb.log({"epoch": epoch+1, 
                   "train_loss": avg_train_loss, 
                   "train_accuracy": train_accuracy, 
                   "val_loss": avg_val_loss, 
                   "val_accuracy": val_accuracy})
        
        print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%".format(
            epoch+1, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy))
        
        # Adjust learning rate
        scheduler.step()
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="CNN Training Script")
    parser.add_argument("--wandb_entity", "-we", default="myname", help="Wandb Entity")
    parser.add_argument("--wandb_project", "-wp", default="myprojectname", help="Wandb Project Name")
    parser.add_argument("--train_path", "-trd", default="C:/Users/navee/OneDrive/Documents/dl/A2/inaturalist_12K/train",type= str , help="TrainDataset")
    parser.add_argument("--test_path", "-ted", default="C:/Users/navee/OneDrive/Documents/dl/A2/inaturalist_12K/val",type= str , help="TestDataset")
    parser.add_argument("--num_epochs", "-e", default=5, type=int, help="Number of epochs")
    parser.add_argument("--batch_size","-b",default=64,type=int, help="Batch size used to train neural network.")
    parser.add_argument("--learning_rate","-lr",default=0.0001,type=float, help="Learning rate used to optimize model parameters")
    parser.add_argument("--lr_decay","-ld",default=0.5,type=float, help="lr decay used by optimizers.")
    parser.add_argument("--activation_conv","-a",default="mish",type=str, choices= ['mish', 'relu', 'silu', 'gelu'])
    parser.add_argument("--batch_norm","-bn",default=True,type=bool, choices= [False, True])
    parser.add_argument("--data_augmentation","-da",default=True,type=bool, choices= [False, True])
    parser.add_argument("--dropout","-d",default=0.3,type=int, help="dropout value")
    parser.add_argument("--num_filters","-nf",default=128,type=int, help="number of filters in the network")
    parser.add_argument("--filter_size","-fs",default=5,type=int, help="size of filters")
    parser.add_argument("--dense_units","-dn",default=256,type=int, help="Dense layer ")
    parser.add_argument("--filter_organization","-fo",default="double",type=str, choices=["double", "same","halve"])

    args = parser.parse_args()
    print(args.train_path)
    # Perform training
    train_model(args)

if __name__ == "__main__":
    main()