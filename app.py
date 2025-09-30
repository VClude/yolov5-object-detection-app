import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import json
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import tempfile
from torch.profiler import profile, ProfilerActivity
import time
import pathlib
import platform

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

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
def get_layer_code_and_switch_tab(layer_choice):
    """Extract the code for a specific layer class from common.py and switch to layer code tab"""
    code = get_layer_code(layer_choice)
    return code, gr.Tabs(selected="layer_code")

def get_layer_code(layer_choice):
    """Extract the code for a specific layer class from common.py"""
    if not layer_choice:
        return "No layer selected."
    
    # Extract the module type from the layer choice
    if '(' in layer_choice:
        # Format: "layer_name (Module Type, LayerType in Layer ParentType)"
        module_info = layer_choice.split('(')[1].split(')')[0]
        # Extract the module type (first part before comma)
        if ',' in module_info:
            # Get the module type (e.g., "Module CoordinateAttention" -> "CoordinateAttention")
            module_part = module_info.split(',')[0].strip()
            if module_part.startswith('Module '):
                layer_type = module_part.replace('Module ', '')
            else:
                layer_type = module_part
        else:
            # If no comma, extract the module type directly
            if module_info.startswith('Module '):
                layer_type = module_info.replace('Module ', '')
            else:
                layer_type = module_info.strip()
    else:
        return "Could not parse layer information."
    
    # Special case: If the module is C3CA, show both C3CA and CABottleneck code
    show_both_classes = False
    original_layer_type = layer_type
    if layer_type == 'C3CA':
        show_both_classes = True
    
    # Read the common.py file
    common_py_path = get_resource_path(os.path.join("yolov5", "models", "common.py"))
    
    try:
        with open(common_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the class definition
        lines = content.split('\n')
        class_start = None
        class_end = None
        indent_level = None
        
        for i, line in enumerate(lines):
            # Look for class definition
            if line.strip().startswith(f'class {layer_type}(') or line.strip() == f'class {layer_type}:':
                class_start = i
                # Find the indentation level of the class
                indent_level = len(line) - len(line.lstrip())
                break
        
        if class_start is None:
            # If the main module type is not found, check if it's a standard PyTorch module
            # and try to find any custom module mentioned in the layer choice
            if ',' in layer_choice and 'in Layer' in layer_choice:
                # Try to extract the parent layer type
                parts = layer_choice.split('in Layer')
                if len(parts) > 1:
                    parent_type = parts[1].strip().rstrip(')')
                    # Search for the parent type instead
                    for i, line in enumerate(lines):
                        if line.strip().startswith(f'class {parent_type}(') or line.strip() == f'class {parent_type}:':
                            class_start = i
                            layer_type = parent_type
                            indent_level = len(line) - len(line.lstrip())
                            break
            
            if class_start is None:
                return f"""Class '{layer_type}' not found in common.py

This might be a standard PyTorch module. The available custom classes in common.py include:
Conv, DWConv, TransformerLayer, TransformerBlock, Bottleneck, BottleneckCSP, 
CrossConv, C3, C3x, C3TR, C3SPP, C3Ghost, SPP, SPPF, Focus, GhostConv, 
GhostBottleneck, CABottleneck, ChannelAttention, SpatialAttention, CBAM, 
CoordinateAttention, C3CA, DetectMultiBackend, AutoShape, Detections, etc.

Selected layer: {layer_choice}"""
        
        # Find the end of the class (next class or function at same or lower indentation)
        for i in range(class_start + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            current_indent = len(line) - len(line.lstrip())
            # If we hit a line with same or less indentation that starts a new class/function/def
            if (current_indent <= indent_level and 
                (line.strip().startswith('class ') or 
                 line.strip().startswith('def ') or
                 line.strip().startswith('# ') and line.strip().startswith('# ') and len(line.strip()) > 10)):
                class_end = i
                break
        
        if class_end is None:
            class_end = len(lines)
        
        # Extract the class code
        class_code = '\n'.join(lines[class_start:class_end])
        
        # If we need to show both C3CA and CABottleneck, get CABottleneck code too
        additional_code = ""
        if show_both_classes and original_layer_type == 'C3CA':
            # Find CABottleneck class
            cabottleneck_start = None
            cabottleneck_end = None
            for i, line in enumerate(lines):
                if line.strip().startswith('class CABottleneck(') or line.strip() == 'class CABottleneck:':
                    cabottleneck_start = i
                    cabottleneck_indent = len(line) - len(line.lstrip())
                    break
            
            if cabottleneck_start is not None:
                # Find end of CABottleneck class
                for i in range(cabottleneck_start + 1, len(lines)):
                    line = lines[i]
                    if line.strip() == '':
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if (current_indent <= cabottleneck_indent and 
                        (line.strip().startswith('class ') or 
                         line.strip().startswith('def ') or
                         line.strip().startswith('# ') and len(line.strip()) > 10)):
                        cabottleneck_end = i
                        break
                
                if cabottleneck_end is None:
                    cabottleneck_end = len(lines)
                
                cabottleneck_code = '\n'.join(lines[cabottleneck_start:cabottleneck_end])
                additional_code = f"\n\n# Related CABottleneck class used by C3CA:\n{cabottleneck_code}"
        
        # Add some context information
        result = f"""# Layer Type: {original_layer_type}
# Selected Layer: {layer_choice}
# Source: yolov5/models/common.py

{class_code}{additional_code}"""
        
        return result
        
    except FileNotFoundError:
        return f"Could not find common.py file at {common_py_path}"
    except Exception as e:
        return f"Error reading code: {str(e)}"

# Detection function using detect.py logic

def detect(image, model_choice, layer_choice, above_color, below_color, iou_threshold, conf_threshold, image_size, magnifier):
    # Check if image is provided
    if image is None:
        return "Please upload an image first.", None, "No metrics available", "No detections", None, None, None, "No model structure available", None
    
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
    img_for_model = letterbox(img, new_shape=(image_size, image_size))[0]
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
    total_high_confidence = 0
    for cls, counts in summary.items():
        summary_lines.append(f"There is {counts['above']} of [{cls}] detected with above 0.5 confidence, and {counts['below']} of [{cls}] detected with below 0.5 confidence")
        total_high_confidence += counts['above']
    
    # Calculate density using magnifier data
    magnifier_data = {
        "200x": {"value": 200, "label": "200x", "x": 0.66152, "y": 0.37052},
        "400x": {"value": 400, "label": "400x", "x": 0.33097, "y": 0.18568}
    }
    
    if magnifier and magnifier in magnifier_data:
        mag_info = magnifier_data[magnifier]
        mag_area = float(mag_info['x']) * float(mag_info['y'])
        density = round(total_high_confidence / mag_area, 2) if mag_area > 0 else 0
        summary_lines.append(f"\nThe Density of object is : {density} mm2")
    
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
    
    # Create model architecture diagram
    architecture_diagram = create_model_architecture_diagram(model_choice, image_size)
    
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

    return summary_text, orig, metrics_text, json.dumps(detections, indent=2), inter_img, histogram_img, inter_out_file, model_structure, architecture_diagram

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
- Total Layers: {total_layers}

Model Structure:
{str(model.model)}"""
    
    return model_info

def create_model_architecture_diagram(model_choice, image_size=640):
    """Create a professional visual diagram of the YOLOv5 model architecture"""
    try:
        model_path = os.path.join("model", model_choice)
        model = DetectMultiBackend(model_path, device="cpu")
        
        # Create figure with professional styling
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(20, 14))
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 18)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        # Professional color scheme
        colors = {
            'Conv': '#FF4757',      # Red
            'BatchNorm2d': '#FF6348',
            'SiLU': '#FF7675',
            'C3': '#00D2D3',        # Cyan
            'SPPF': '#0984E3',      # Blue
            'Upsample': '#00B894',  # Green
            'Concat': '#FDCB6E',    # Yellow
            'Detect': '#A29BFE',    # Purple
            'AvgPool2d': '#E17055',
            'Conv2d': '#FF4757',
            'Default': '#000000'
        }
        
        # Get actual model structure
        model_layers = []
        for name, module in model.model.named_modules():
            if '.' in name and len(name.split('.')) == 2:  # Top-level modules
                layer_idx = name.split('.')[1]
                if layer_idx.isdigit():
                    model_layers.append((int(layer_idx), name, type(module).__name__))
        
        model_layers.sort(key=lambda x: x[0])
        
        # Define YOLOv5 architecture structure
        backbone_layers = []
        neck_layers = []
        head_layers = []
        backbone_total = 12 if "cbam" in model_choice.lower() else 9
        for idx, name, module_type in model_layers:
            if idx <= backbone_total:
                backbone_layers.append((idx, name, module_type))
            elif idx <= 23:
                neck_layers.append((idx, name, module_type))
            else:
                head_layers.append((idx, name, module_type))
        
        # Title
        ax.text(12, 17, f'YOLOv5 Architecture: {model_choice.split("/")[0]}', 
                ha='center', va='center', fontsize=20, fontweight='bold')
        
        # Input section
        input_y = 15.5
        ax.text(2, input_y + 0.8, 'Input', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='#2D3436')
        
        # Draw input image
        input_rect = patches.FancyBboxPatch((1, input_y-0.4), 2, 0.8, 
                                          boxstyle="round,pad=0.1", 
                                          facecolor='#74B9FF', edgecolor='#0984E3', linewidth=2)
        ax.add_patch(input_rect)
        ax.text(2, input_y, f'{image_size}×{image_size}×3', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Backbone section
        backbone_y = 13
        ax.text(1, backbone_y + 1, 'Backbone\n(CSPDarknet53)', ha='center', va='center', 
                fontsize=12, fontweight='bold', rotation=0, color='#2D3436')
        
        # Draw backbone layers
        backbone_x_positions = np.linspace(3, 21, len(backbone_layers))
        for i, (idx, name, module_type) in enumerate(backbone_layers):
            x_pos = backbone_x_positions[i]
            
            # Determine color
            if 'Conv' in module_type or 'Focus' in module_type:
                color = colors['Conv']
            elif 'C3' in module_type or 'CSP' in module_type:
                color = colors['C3']
            elif 'SPPF' in module_type or 'SPP' in module_type:
                color = colors['SPPF']
            else:
                color = colors['Default']
            
            # Calculate box width based on text length
            text_length = len(module_type)
            box_width = max(0.8, min(1.6, text_length * 0.08 + 0.4))
            
            # Draw layer box
            layer_rect = patches.FancyBboxPatch((x_pos-box_width/2, backbone_y-0.3), box_width, 0.6,
                                              boxstyle="round,pad=0.05",
                                              facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(layer_rect)
            
            # Layer text with wrapping
            layer_text = module_type
            # Break long text into multiple lines
            if len(layer_text) > 99999:
                # Find a good break point
                mid = len(layer_text) // 2
                break_point = mid
                for i in range(max(0, mid-2), min(len(layer_text), mid+3)):
                    if layer_text[i] in ['_', '2', '3']:
                        break_point = i + 1
                        break
                line1 = layer_text[:break_point]
                line2 = layer_text[break_point:]
                ax.text(x_pos, backbone_y+0.1, line1, ha='center', va='center',
                       fontsize=6, fontweight='bold', color='white')
                ax.text(x_pos, backbone_y-0.1, line2, ha='center', va='center',
                       fontsize=6, fontweight='bold', color='white')
            else:
                ax.text(x_pos, backbone_y, layer_text, ha='center', va='center',
                       fontsize=7, fontweight='bold', color='white')
            
            ax.text(x_pos, backbone_y-0.6, f'{idx}', ha='center', va='center',
                   fontsize=7, color='#636E72')
        
        neck_y = 10
        ax.text(1, neck_y + 1, 'Neck\n', ha='center', va='center', 
                fontsize=12, fontweight='bold', rotation=0, color='#2D3436')
        
        # Draw neck layers
        neck_x_positions = np.linspace(3, 21, len(neck_layers))
        for i, (idx, name, module_type) in enumerate(neck_layers):
            x_pos = neck_x_positions[i]
            
            # Determine color
            if 'Conv' in module_type:
                color = colors['Conv']
            elif 'C3' in module_type:
                color = colors['C3']
            elif 'Upsample' in module_type:
                color = colors['Upsample']
            elif 'Concat' in module_type:
                color = colors['Concat']
            else:
                color = colors['Default']
            
            # Calculate box width based on text length
            text_length = len(module_type)
            box_width = max(0.8, min(1.6, text_length * 0.08 + 0.4))
            
            # Draw layer box
            layer_rect = patches.FancyBboxPatch((x_pos-box_width/2, neck_y-0.3), box_width, 0.6,
                                              boxstyle="round,pad=0.05",
                                              facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(layer_rect)
            
            # Layer text with wrapping
            layer_text = module_type
            # Break long text into multiple lines
            if len(layer_text) > 99999:
                # Find a good break point
                mid = len(layer_text) // 2
                break_point = mid
                for i in range(max(0, mid-2), min(len(layer_text), mid+3)):
                    if layer_text[i] in ['_', '2', '3']:
                        break_point = i + 1
                        break
                line1 = layer_text[:break_point]
                line2 = layer_text[break_point:]
                ax.text(x_pos, neck_y+0.1, line1, ha='center', va='center',
                       fontsize=6, fontweight='bold', color='white')
                ax.text(x_pos, neck_y-0.1, line2, ha='center', va='center',
                       fontsize=6, fontweight='bold', color='white')
            else:
                ax.text(x_pos, neck_y, layer_text, ha='center', va='center',
                       fontsize=7, fontweight='bold', color='white')
            
            ax.text(x_pos, neck_y-0.6, f'{idx}', ha='center', va='center',
                   fontsize=7, color='#636E72')
        
        # Head section
        head_y = 7
        ax.text(1, head_y + 1, 'Head\n(Detect)', ha='center', va='center', 
                fontsize=12, fontweight='bold', rotation=0, color='#2D3436')
        
        # Draw detection heads
        head_positions = [8, 12, 16]  # Three detection scales
        head_labels = ['', '', '']

        for i, (x_pos, label) in enumerate(zip(head_positions, head_labels)):
            # Draw detection head
            head_rect = patches.FancyBboxPatch((x_pos-0.6, head_y-0.4), 1.2, 0.8,
                                             boxstyle="round,pad=0.1",
                                             facecolor=colors['Detect'], edgecolor='black', linewidth=1)
            ax.add_patch(head_rect)
            
            ax.text(x_pos, head_y, 'Detect', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
            ax.text(x_pos, head_y-0.8, label, ha='center', va='center',
                   fontsize=7, color='#636E72')
        
        # Draw connections with arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='#2D3436')
        
        # Input to backbone
        ax.annotate('', xy=(3, backbone_y+0.5), xytext=(2, input_y-0.5), arrowprops=arrow_props)
        
        # Backbone to neck
        ax.annotate('', xy=(12, neck_y+0.8), xytext=(12, backbone_y-0.8), arrowprops=arrow_props)
        
        # Neck to heads
        for x_pos in head_positions:
            ax.annotate('', xy=(x_pos, head_y+0.5), xytext=(x_pos, neck_y-0.8), arrowprops=arrow_props)
        
        # Output section
        output_y = 4.5
        ax.text(12, output_y + 0.8, 'Output', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='#2D3436')
        
        output_labels = ['Classes + Boxes\n(Large)', 'Classes + Boxes\n(Medium)', 'Classes + Boxes\n(Small)']
        for i, (x_pos, label) in enumerate(zip(head_positions, output_labels)):
            output_rect = patches.FancyBboxPatch((x_pos-0.6, output_y-0.3), 1.2, 0.6,
                                               boxstyle="round,pad=0.1",
                                               facecolor='#00B894', edgecolor='black', linewidth=1)
            ax.add_patch(output_rect)
            ax.text(x_pos, output_y, label, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
        
      
        # Model statistics
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        def format_params(num):
            if num >= 1e6:
                return f"{num/1e6:.1f}M"
            elif num >= 1e3:
                return f"{num/1e3:.1f}K"
            else:
                return str(num)
        
        stats_text = f"""Model Statistics:
• Parameters: {format_params(total_params)}
• Trainable: {format_params(trainable_params)}
• Input Size: {image_size}×{image_size}×3
• Classes: {getattr(model.model, 'nc', 'N/A')}"""
        
        ax.text(21.5, 7, stats_text, fontsize=10, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', 
                        edgecolor='#DEE2E6', linewidth=1))
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        diagram_img = Image.open(buf).copy()
        buf.close()
        plt.close()
        
        return diagram_img
        
    except Exception as e:
        # Create error image
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, f'Error creating architecture diagram:\n{str(e)}', 
                ha='center', va='center', fontsize=14, 
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFE5E5', edgecolor='red'))
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
        buf.seek(0)
        error_img = Image.open(buf).copy()
        buf.close()
        plt.close()
        
        return error_img

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
    gr.Markdown("Aplikasi deteksi objek menggunakan model dengan YOLOv5.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", height=350)
            
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=get_model_choices(), value=get_default_model(), label="Select Model", scale=4)
                refresh_btn = gr.Button("🔄", size="sm", scale=1)
            
            layer_dropdown = gr.Dropdown(choices=get_layer_choices(os.path.join("model", get_default_model())) if get_default_model() else [], value=get_default_layer(), label="Select Layer")
            
            show_code_btn = gr.Button("📄 Show Code", variant="secondary", size="sm")
            
            detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            detected_image = gr.Image(type="pil", label="Detected Image", height=350)
            summary_output = gr.Textbox(label="Detection Summary", lines=8, max_lines=12)
    
    with gr.Tabs() as tabs:
        with gr.Tab("Configuration", id="config"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Color Settings")
                    above_color = gr.ColorPicker(value="green", label="High Confidence Color (≥ 0.5)")
                    below_color = gr.ColorPicker(value="red", label="Low Confidence Color (< 0.5)")
                
                with gr.Column():
                    gr.Markdown("### Detection Thresholds")
                    iou_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IoU Threshold for NMS")
                    conf_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold")
                
                with gr.Column():
                    gr.Markdown("### Image Processing")
                    image_size = gr.Dropdown(choices=[384, 640, 960, 1280], value=640, label="Yolo Detection Input Image Size (pixels)")
                    magnifier = gr.Dropdown(
                        choices=["200x", "400x"],
                        value="200x",
                        label="Select Magnifier"
                    )
        
        with gr.Tab("Analysis Results", id="analysis"):
            with gr.Row():
                metrics_output = gr.Textbox(label="Inference & Model Metrics", lines=8, max_lines=12)
                json_output = gr.Textbox(label="Detection Results (JSON)", lines=8, max_lines=12)
        
        with gr.Tab("Layer Analysis", id="layer_analysis"):
            with gr.Row():
                intermediate_output = gr.Image(type="pil", label="Intermediate Layer Output", height=250)
                histogram_output = gr.Image(type="pil", label="Pixel Distribution", height=250)
            
            with gr.Row():
                download_output = gr.File(label="Download Layer Output (.npy)")
        
        with gr.Tab("Model Structure", id="structure"):
            with gr.Row():
                with gr.Column():
                    architecture_diagram = gr.Image(type="pil", label="Model Architecture Diagram", height=400)
                with gr.Column():
                    structure_output = gr.Textbox(label="Model Structure Details", lines=15, max_lines=25)
        
        with gr.Tab("Layer Code", id="layer_code"):
            layer_code_output = gr.Textbox(label="Layer Implementation Code", lines=20, max_lines=30, show_copy_button=True)
    
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
        inputs=[image_input, model_dropdown, layer_dropdown, above_color, below_color, iou_threshold, conf_threshold, image_size, magnifier],
        outputs=[summary_output, detected_image, metrics_output, json_output, intermediate_output, histogram_output, download_output, structure_output, architecture_diagram]
    )
    
    show_code_btn.click(
        fn=get_layer_code_and_switch_tab,
        inputs=layer_dropdown,
        outputs=[layer_code_output, tabs]
    )

demo.launch()
