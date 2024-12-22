import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

# Cihaz ayarı (CPU veya GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model ve ağırlıkların yüklenmesi
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # En güncel varsayılan ağırlıkları kullan
model = fasterrcnn_resnet50_fpn(weights=weights)  # Modeli ağırlıklarla yükle
model = model.to(device)  # Modeli cihazda çalışacak şekilde taşı
model.eval()  # Modeli değerlendirme moduna al

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Çözünürlüğü düşür
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
detection_interval = 1  # Her 5 karede bir tespit yap

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    image_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    # Perform detection
    with torch.no_grad():
        outputs = model(image_tensor)

    # Draw bounding boxes
    for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label}, Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Faster R-CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
