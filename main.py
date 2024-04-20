import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms

import load_data


class VehicleDataset(Dataset):
    def __init__(self, images, labels, bboxes):
        self.images = images
        self.labels = labels
        self.bboxes = bboxes
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        bbox = self.bboxes[idx]
        cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        image_tensor = self.transforms(cropped_image)
        return image_tensor, torch.tensor([label])


class AnglePredictor(nn.Module):
    def __init__(self):
        super(AnglePredictor, self).__init__()
        self.resnet_model = models.resnet50(pretrained=True)

        # Freeze all layers first
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        # Unfreeze the last 15 layers
        num_layers = len(list(self.resnet_model.children()))
        layers_to_unfreeze = list(self.resnet_model.children())[num_layers - 15:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        # Modify the ResNet model to not include the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet_model.children())[:-1])

        # Expanded regression head
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Outputting a single value for angle
        )

    def forward(self, pixel_values):
        features = self.features(pixel_values)
        angle = self.regression_head(features)
        return angle


def get_data_loaders(train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes, batch_size=4):
    train_dataset = VehicleDataset(train_images, train_labels, train_bboxes)
    test_dataset = VehicleDataset(test_images, test_labels, test_bboxes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
    average_loss = total_loss / len(test_loader)
    print(f"Test MAE Loss: {average_loss}")
    return average_loss


def train_model(model, train_loader, test_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, MSE Loss: {running_loss / len(train_loader)}")
        evaluate(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", default=4, type=int, help="batch size")
    parser.add_argument("--epoch", default=50, type=int)

    args = parser.parse_args()
    print(args)

    train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes = load_data.load_data()

    train_loader, test_loader = get_data_loaders(
        train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes,
        args.bz,
    )

    model = AnglePredictor()
    train_model(model, train_loader, test_loader, num_epochs=args.epoch)


if __name__ == "__main__":
    main()
