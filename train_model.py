import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataset import CustomDataset

# Cihaz ayarı (CPU veya GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Görüntü dizini ve anotasyonlar
img_dir = "E:\\görüntü işleme\\yolo\\dataset\\combined"
annotations = [
    {"file_name": "image1.jpg", "boxes": [[100, 200, 300, 400]], "labels": [1]},
    {"file_name": "image2.jpg", "boxes": [[50, 50, 200, 300]], "labels": [2]},
]

# Veri seti ve veri yükleyici oluşturma
dataset = CustomDataset(img_dir, annotations)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Önceden eğitilmiş Faster R-CNN modelini yükleme
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)

# Eğitim döngüsü
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # Epoch sayısı
    for images, targets in data_loader:
        print(f"Processing batch with {len(images)} images")

        # Verileri uygun cihaza taşıma
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # İleri geçiş
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Geri yayılım
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}], Loss: {losses.item()}")