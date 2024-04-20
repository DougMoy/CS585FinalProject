import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import DetrForObjectDetection, DetrImageProcessor

import load_data


class VehicleDataset(Dataset):
    def __init__(self, images, labels, bboxes, processor):
        self.images = images
        self.labels = labels
        self.bboxes = bboxes
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        bbox = self.bboxes[idx]

        cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        resized_image = cropped_image.resize((224, 224))

        inputs = self.processor(images=resized_image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), torch.tensor([label])


def get_data_loaders(
        train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes,
        processor, batch_size=4,
):
    train_dataset = VehicleDataset(train_images, train_labels, train_bboxes, processor)
    test_dataset = VehicleDataset(test_images, test_labels, test_bboxes, processor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model, test_loader, criterion, device):
    model.eval()
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

    for epoch in range(num_epochs):
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
        evaluate(model, test_loader, criterion, device)


class AnglePredictor(nn.Module):
    def __init__(self, detr_model, processor):
        super(AnglePredictor, self).__init__()
        self.detr_model = detr_model
        self.processor = processor
        self.regression_head = nn.Linear(256, 1)  # assuming feature dimension of 256

    def forward(self, pixel_values):
        outputs = self.detr_model(pixel_values)
        # Using mean pooling of features from all detected objects
        features = torch.mean(outputs.last_hidden_state, dim=1)
        angle = self.regression_head(features)
        return angle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", default=4, type=int, help="batch size")

    args = parser.parse_args()
    print(args)

    detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

    train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes = load_data.load_data()
    train_loader, test_loader = get_data_loaders(
        train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes,
        processor, args.bz,
    )

    model = AnglePredictor(detr_model, processor)
    train_model(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
