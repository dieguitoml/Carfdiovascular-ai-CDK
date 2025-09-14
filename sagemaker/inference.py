import os, io, json, base64
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train_sagemaker import ECGImageCNN  # misma red que en training

def _build_tf(img_h, img_w):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_h, img_w), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])

def model_fn(model_dir):
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", (450, 1500))
    model = ECGImageCNN(in_channels=1, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model._classes = classes
    model._tf = _build_tf(*img_size)
    return model

def input_fn(request_body, request_content_type="application/json"):
    assert request_content_type == "application/json"
    data = json.loads(request_body or "{}")
    if "image_base64" in data:
        img = Image.open(io.BytesIO(base64.b64decode(data["image_base64"])))
    elif "filepath" in data:
        img = Image.open(data["filepath"])
    elif "pixels" in data:
        import numpy as np
        arr = np.array(data["pixels"]).astype("float32")
        img = Image.fromarray(arr)
    else:
        raise ValueError("Se requiere 'image_base64', 'filepath' o 'pixels'.")
    return img

def predict_fn(img: Image.Image, model):
    x = model._tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        idx = int(logits.argmax(1))
    return {"label": model._classes[idx],
            "probs": {c: float(p) for c,p in zip(model._classes, probs)}}

def output_fn(pred, accept="application/json"):
    return json.dumps(pred), "application/json"
