import argparse
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
        self.resnet_model = nn.Sequential(*list(self.resnet_model.children())[:-1])
        self.regression_head = nn.Linear(2048, 1)

    def forward(self, pixel_values):
        features = self.resnet_model(pixel_values)
        features = features.view(features.size(0), -1)
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
    print(f"Test Loss: {average_loss}")
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

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        evaluate(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", default=4, type=int, help="batch size")
    parser.add_argument("--epoch", default=100, type=int)

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
