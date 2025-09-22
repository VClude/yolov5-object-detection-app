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
    model_dir = 'model'
    choices = []
    for name in os.listdir(model_dir):
        weights_path = os.path.join(model_dir, name, "weights", "best.pt")
        if os.path.isfile(weights_path):
            choices.append(f"{name}/weights/best.pt")
    return choices

# Custom class names
class_names = ["km"]

# Helper to get layer names with type and backbone/head info

def get_layer_choices(model_path):
    model = DetectMultiBackend(model_path, device="cpu")
    layers = []
    # Find head start index (Detect/Segment)
    head_types = (getattr(model.model, 'Detect', None), getattr(model.model, 'Segment', None))
    head_indices = []
    for i, m in enumerate(model.model.modules()):
        if type(m).__name__ in ['Detect', 'Segment']:
            head_indices.append(i)
    # Fallback: last Detect/Segment is head
    last_head_idx = head_indices[-1] if head_indices else None
    for idx, (name, module) in enumerate(model.model.named_modules()):
        if len(list(module.children())) == 0:
            layer_type = type(module).__name__
            # Determine backbone/head
            if last_head_idx is not None and idx >= last_head_idx:
                part = 'head'
            else:
                part = 'backbone'
            layers.append(f"{name} ({layer_type}, {part})")
    return layers

# Detection function using detect.py logic

def detect(image, model_choice, layer_choice):
    # Resize input image to square before processing
    img_resized = image.resize((640, 640))
    img = np.array(img_resized)
    model_path = os.path.join("model", model_choice)
    device = "cpu"
    model = DetectMultiBackend(model_path, device=device)
    stride, names, pt = model.stride, class_names, model.pt
    imgsz = (640, 640)
    model.eval()

    layer_name = layer_choice.split(' (')[0] if layer_choice else None

    # Preprocess for model
    img_for_model, ratio, pad = letterbox(img, new_shape=imgsz, auto=pt)
    img_for_model = img_for_model.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_for_model).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    intermediate = {}
    def hook_fn(module, input, output):
        intermediate['output'] = output.detach().cpu().numpy()
    for name, module in model.model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
            break

    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000)

    draw = ImageDraw.Draw(img_resized)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=20)
    except:
        font = ImageFont.load_default()

    detections = []
    summary = {}
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img_resized.size).round()
            for *xyxy, conf, cls in det:
                cls_id = int(cls.item())
                label = f"{names[cls_id]}: {conf:.2f}"
                color = "green" if conf >= 0.5 else "red"
                draw.rectangle(xyxy, outline=color, width=4)
                draw.text((xyxy[0], xyxy[1] - 25), label, fill=color, font=font)
                detections.append({
                    "class": names[cls_id],
                    "confidence": round(float(conf), 3),
                    "bbox": [round(float(x), 1) for x in xyxy]
                })
                conf_key = 'above' if conf >= 0.5 else 'below'
                if names[cls_id] not in summary:
                    summary[names[cls_id]] = {'above': 0, 'below': 0}
                summary[names[cls_id]][conf_key] += 1

    summary_lines = []
    for cls, counts in summary.items():
        summary_lines.append(f"There is {counts['above']} of [{cls}] detected with above 0.5 confidence, and {counts['below']} of [{cls}] detected with below 0.5 confidence")
    summary_text = "\n".join(summary_lines) if summary_lines else "No objects detected."

    inter_out = intermediate.get('output')
    if inter_out is not None:
        arr = inter_out
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr[0] if arr.shape[0] > 0 else arr
        arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255
        inter_img = Image.fromarray(arr.astype(np.uint8))
    else:
        inter_img = None

    return img_resized, json.dumps(detections, indent=2), inter_img, get_model_structure(model_choice), summary_text, img_resized

def get_model_structure(model_choice):
    model_path = os.path.join("model", model_choice)
    model = DetectMultiBackend(model_path, device="cpu")
    return str(model.model)

# Gradio Interface with model and layer selection

def get_default_model():
    choices = get_model_choices()
    return choices[0] if choices else None

def get_default_layer():
    model_choice = get_default_model()
    if model_choice:
        layers = get_layer_choices(os.path.join("model", model_choice))
        return layers[0] if layers else None
    return None

demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(choices=get_model_choices(), value=get_default_model(), label="Select Model"),
        gr.Dropdown(choices=get_layer_choices(os.path.join("model", get_default_model())) if get_default_model() else [], value=get_default_layer(), label="Select Layer")
    ],
    outputs=[
        gr.Image(type="pil", label="Detected Image"),
        gr.Textbox(label="Detection Results (JSON)"),
        gr.Image(type="pil", label="Intermediate Layer Output"),
        gr.Textbox(label="Model Structure"),
        gr.Textbox(label="Detection Summary"),
    ],
    title="YOLOv5 Object Detection",
    description="Upload an image, select a model and layer to view detection, intermediate output, model structure, summary, and preprocessed image."
)

demo.launch()
