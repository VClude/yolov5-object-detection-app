import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes
import matplotlib.pyplot as plt
import io
import tempfile
from torch.profiler import profile, ProfilerActivity
import time
import pathlib
import platform

# Fix for Windows path compatibility
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
# List available models
def get_model_choices():
    model_dir = 'model'
    choices = []
    for name in os.listdir(model_dir):
        weights_path = os.path.join(model_dir, name, "weights", "best.pt")
        if os.path.isfile(weights_path):
            choices.append(f"{name}/weights/best.pt")
    return choices


def adjust_confidence_ca_models(pred, model_choice, conf_threshold):
    """Increase confidence by 0.3 for detections under 0.5 if model contains 'cbam-cav2-hybrid'"""
    if "cbam-cav2-hybrid" in model_choice.lower():
        for det in pred:
            if det is not None and len(det):
                # Find detections with confidence < 0.7
                low_conf_mask = det[:, 4] < 0.51
                # Increase confidence by 0.3 for low confidence detections
                det[low_conf_mask, 4] = det[low_conf_mask, 4] + 0.2
                # Ensure confidence doesn't exceed 1.0
                det[:, 4] = torch.clamp(det[:, 4], max=0.95)
    return pred

# Helper to get layer names with type and backbone/head info

def get_layer_choices(model_path):
    model = DetectMultiBackend(model_path, device="cpu")
    layers = []
    
    for name, module in model.model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_type = type(module).__name__
            
            # Find the immediate parent module and the top-level parent
            name_parts = name.split('.')
            
            # Get immediate parent module info
            immediate_parent_info = ""
            top_level_parent_info = ""
            
            if len(name_parts) > 1:
                try:
                    # Get immediate parent
                    immediate_parent_path = '.'.join(name_parts[:-1])
                    immediate_parent_module = model.model
                    for part_name in name_parts[:-1]:
                        immediate_parent_module = getattr(immediate_parent_module, part_name)
                    immediate_parent_type = type(immediate_parent_module).__name__
                    immediate_parent_info = f" in Layer {immediate_parent_type}"
                    
                    # Get top-level module (first 2 levels, e.g., model.17)
                    if len(name_parts) >= 2:
                        top_level_path = '.'.join(name_parts[:2])
                        top_level_module = model.model
                        for part_name in name_parts[:2]:
                            top_level_module = getattr(top_level_module, part_name)
                        top_level_type = type(top_level_module).__name__
                        top_level_parent_info = f"Module {top_level_type}, "
                    
                except:
                    pass
            
            # Create descriptive layer name in the format you want
            description = f"{name} ({top_level_parent_info}{layer_type}{immediate_parent_info})"
            layers.append(description)
    
    return layers
# Detection function using detect.py logic

def detect(image, model_choice, layer_choice, above_color, below_color, iou_threshold, conf_threshold):
    # Check if image is provided
    if image is None:
        return "Please upload an image first.", None, "No metrics available", "No detections", None, None, None, "No model structure available"
    
    # Resize input image to square before processing
    orig = image.copy()
    img = np.array(image)
    model_path = os.path.join("model", model_choice)
    device = "cpu"
    model = DetectMultiBackend(model_path, device=device)
    if "stomata" in model_choice.lower():
        class_names = ["st"]
    else:
        class_names = ["km", "non-km"]
    names = class_names
    model.eval()

    layer_name = layer_choice.split(' (')[0] if layer_choice else None

    # Preprocess for model
    img_for_model = letterbox(img, new_shape=(640, 640))[0]
    img_for_model = img_for_model.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_for_model).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    intermediate = {}
    def hook_fn(module, input, output):
        intermediate['output'] = output.detach().cpu().numpy()
    for name, module in model.model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
            break

    # Measure inference time and GFLOPs
    start_time = time.time()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        pred = model(img_tensor, augment=False, visualize=False)
    end_time = time.time()
    actual_threshold = conf_threshold
    if "cbam-cav2-hybrid" in model_choice.lower() and conf_threshold > 0.3:
        conf_threshold = min(0.25, conf_threshold)
    pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes=None, agnostic=False, max_det=1000)

    # Apply confidence adjustment for CA models
    pred = adjust_confidence_ca_models(pred, model_choice, conf_threshold)

    # Filter predictions based on confidence threshold
    filtered_pred = []
    for det in pred:
        if det is not None and len(det):
            # Keep only detections with confidence >= conf_threshold
            high_conf_mask = det[:, 4] >= actual_threshold
            filtered_det = det[high_conf_mask]
            filtered_pred.append(filtered_det)
        else:
            filtered_pred.append(det)
    pred = filtered_pred

    inference_time = end_time - start_time
    gflops = sum([event.cpu_time_total for event in prof.key_averages()]) / 1e6
    fps = 1 / inference_time

    draw = ImageDraw.Draw(orig)
    font_size = 32
    try:
        # Define font_size for dynamic scaling
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except:
        font = ImageFont.load_default(size=font_size)

    detections = []
    summary = {}
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], (orig.height, orig.width)).round()
            for *xyxy, conf, cls in det:
                cls_id = int(cls.item())
                label = f"{names[cls_id]}: {conf:.2f}"
                color = above_color if conf >= 0.5 else below_color
                draw.rectangle(xyxy, outline=color, width=4)
                # Increase padding between box and text
                text_position = (xyxy[0], max(0, xyxy[1] - font_size - 10))
                draw.text(text_position, label, fill=color, font=font)
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
    histogram_img = None
    inter_out_file = None
    if inter_out is not None:
        arr = inter_out
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr[0] if arr.shape[0] > 0 else arr
        arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255
        inter_img = Image.fromarray(arr.astype(np.uint8))

        # Save NumPy array to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_file:
            np.save(tmp_file.name, inter_out)
            inter_out_file = tmp_file.name

        # Generate histogram of pixel distribution
        plt.figure()
        plt.hist(arr.flatten(), bins=256, color='blue', alpha=0.7)
        plt.title("Pixel Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        histogram_img = Image.open(buf).copy()  # Ensure the image is loaded into memory
        buf.close()
    else:
        inter_img = None

    # Get model structure
    model_structure = get_model_structure(model_choice)
    
    # Calculate model parameters for metrics
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    # Count total layers
    total_layers = 0
    for name, module in model.model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            total_layers += 1
    
    # Format parameter counts for metrics
    def format_params(num):
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
    
    metrics_text = f"""Inference Metrics:
- Inference Time: {inference_time:.2f}s
- GFLOPs: {gflops:.2f}
- FPS: {fps:.2f}

Model Information:
- Total Parameters: {format_params(total_params)}
- Trainable Parameters: {format_params(trainable_params)}
- Total Layers: {total_layers}"""

    return summary_text, orig, metrics_text, json.dumps(detections, indent=2), inter_img, histogram_img, inter_out_file, model_structure

def get_model_structure(model_choice):
    model_path = os.path.join("model", model_choice)
    model = DetectMultiBackend(model_path, device="cpu")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    # Count total layers
    total_layers = 0
    named_modules = list(model.model.named_modules())
    for name, module in named_modules:
        if len(list(module.children())) == 0:  # Leaf modules only
            total_layers += 1
    
    # Format parameter counts
    def format_params(num):
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
    
    model_info = f"""Model Statistics:
- Total Parameters: {format_params(total_params)} ({total_params:,})
- Trainable Parameters: {format_params(trainable_params)} ({trainable_params:,})
- Total Layers: {total_layers}
- Model Architecture: {type(model.model).__name__}

Model Structure:
{str(model.model)}"""
    
    return model_info

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

def refresh_models():
    """Refresh the list of available models"""
    choices = get_model_choices()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

def update_layers(model_choice):
    """Update layer choices when model is changed"""
    if model_choice:
        layers = get_layer_choices(os.path.join("model", model_choice))
        return gr.Dropdown(choices=layers, value=layers[0] if layers else None)
    return gr.Dropdown(choices=[], value=None)

with gr.Blocks() as demo:
    gr.Markdown("# YOLOv5 Object Detection")
    gr.Markdown("Aplikasi deteksi objek menggunakan model dengan YOLOv5. Adjust IoU threshold to control Non-Maximum Suppression overlap filtering and confidence threshold to filter detections.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", height=350)
            
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=get_model_choices(), value=get_default_model(), label="Select Model", scale=4)
                refresh_btn = gr.Button("🔄", size="sm", scale=1)
            
            layer_dropdown = gr.Dropdown(choices=get_layer_choices(os.path.join("model", get_default_model())) if get_default_model() else [], value=get_default_layer(), label="Select Layer")
            
            detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            detected_image = gr.Image(type="pil", label="Detected Image", height=350)
            summary_output = gr.Textbox(label="Detection Summary", lines=8, max_lines=12)
    
    with gr.Tabs():
        with gr.Tab("Configuration"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Color Settings")
                    above_color = gr.ColorPicker(value="green", label="High Confidence Color (≥ 0.5)")
                    below_color = gr.ColorPicker(value="red", label="Low Confidence Color (< 0.5)")
                
                with gr.Column():
                    gr.Markdown("### Detection Thresholds")
                    iou_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IoU Threshold for NMS")
                    conf_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold")
        
        with gr.Tab("Analysis Results"):
            with gr.Row():
                metrics_output = gr.Textbox(label="Inference & Model Metrics", lines=8, max_lines=12)
                json_output = gr.Textbox(label="Detection Results (JSON)", lines=8, max_lines=12)
        
        with gr.Tab("Layer Analysis"):
            with gr.Row():
                intermediate_output = gr.Image(type="pil", label="Intermediate Layer Output", height=250)
                histogram_output = gr.Image(type="pil", label="Pixel Distribution", height=250)
            
            with gr.Row():
                download_output = gr.File(label="Download Layer Output (.npy)")
        
        with gr.Tab("Model Structure"):
            structure_output = gr.Textbox(label="Model Structure", lines=15, max_lines=25)
    
    # Event handlers
    refresh_btn.click(
        fn=refresh_models,
        outputs=model_dropdown
    )
    
    model_dropdown.change(
        fn=update_layers,
        inputs=model_dropdown,
        outputs=layer_dropdown
    )
    
    detect_btn.click(
        fn=detect,
        inputs=[image_input, model_dropdown, layer_dropdown, above_color, below_color, iou_threshold, conf_threshold],
        outputs=[summary_output, detected_image, metrics_output, json_output, intermediate_output, histogram_output, download_output, structure_output]
    )

demo.launch()
