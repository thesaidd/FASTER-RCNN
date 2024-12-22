import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations):
        self.img_dir = img_dir
        self.annotations = annotations
        # Define a transform to convert PIL images to tensors
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, annotation["file_name"])
        print(f"Constructed image path: {img_path}")

        # Load the image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        print("Image loaded successfully.")

        # Apply the transformation to convert to tensor
        image = self.transform(image)

        boxes = annotation["boxes"]
        labels = annotation["labels"]

        # Convert boxes and labels to tensors
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image, target
