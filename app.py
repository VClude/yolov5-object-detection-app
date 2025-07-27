import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes
# List available models
def get_model_choices():
    model_dir = "model"
    choices = []
    for name in os.listdir(model_dir):
        path = os.path.join(model_dir, name, "best.pt")
        if os.path.isfile(path):
            choices.append(f"{name}/best.pt")
    return choices

# Custom class names
class_names = ["km"]

# Detection function with model selection
def detect(image, model_choice):
    orig = image.copy()
    img = np.array(image)

    # Load selected model
    model_path = os.path.join("model", model_choice)
    model = DetectMultiBackend(model_path, device="cpu")
    model.eval()

    # Preprocess
    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized.transpose((2, 0, 1))  # HWC to CHW
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Inference
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45)

    # Draw results
    draw = ImageDraw.Draw(orig)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=20)
    except:
        font = ImageFont.load_default()

    detections = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], orig.size).round()
            for *xyxy, conf, cls in det:
                cls_id = int(cls.item())
                label = f"{class_names[cls_id]}: {conf:.2f}"
                draw.rectangle(xyxy, outline="red", width=4)
                draw.text((xyxy[0], xyxy[1] - 25), label, fill="red", font=font)
                detections.append({
                    "class": class_names[cls_id],
                    "confidence": round(float(conf), 3),
                    "bbox": [round(float(x), 1) for x in xyxy]
                })

    return orig, json.dumps(detections, indent=2)

# Gradio Interface with model selection
demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(choices=get_model_choices(), value=get_model_choices()[0] if get_model_choices() else None, label="Select Model")
    ],
    outputs=[
        gr.Image(type="pil", label="Detected Image"),
        gr.Textbox(label="Detection Results (JSON)")
    ],
    title="YOLOv5 Object Detection",
    description="Upload an image and select a model to detect objects using your YOLOv5 model."
)

demo.launch()
