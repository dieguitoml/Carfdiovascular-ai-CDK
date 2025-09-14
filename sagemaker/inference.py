# sagemaker/inference.py
import os, io, json, base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- MISMA ARQUITECTURA QUE ENTRENASTE ---
class ECGImageCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.classifier(x)

# --- HELPERS ---
def _build_tf(img_h, img_w):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_h, img_w), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])

# SageMaker llama a esto para cargar el modelo (una vez por contenedor)
def model_fn(model_dir):
    # Carga checkpoint
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Clases e input size desde ckpt o meta.json
    classes = ckpt.get("classes")
    img_size = ckpt.get("img_size")
    if classes is None or img_size is None:
        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        classes = meta["classes"]
        img_size = tuple(meta["img_size"])

    model = ECGImageCNN(in_channels=1, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    model._classes = classes
    model._tf = _build_tf(*img_size)
    return model

# Convierte el body a PIL.Image
def input_fn(request_body, request_content_type="application/json"):
    if request_content_type != "application/json":
        raise ValueError("Content-Type debe ser application/json")
    data = json.loads(request_body or "{}")

    if "image_base64" in data:
        raw = base64.b64decode(data["image_base64"])
        return Image.open(io.BytesIO(raw))
    elif "filepath" in data:
        return Image.open(data["filepath"])
    elif "pixels" in data:
        import numpy as np
        arr = np.array(data["pixels"]).astype("float32")
        return Image.fromarray(arr)
    else:
        raise ValueError("Incluye 'image_base64', 'filepath' o 'pixels' en el JSON.")

# Inferencia pura (sin side-effects)
def predict_fn(img: Image.Image, model):
    x = model._tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        idx = int(logits.argmax(1))
    return {
        "label": model._classes[idx],
        "probs": {c: float(p) for c, p in zip(model._classes, probs)}
    }

def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
