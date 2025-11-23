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
from pathlib import Path

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Fix for Windows path compatibility
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

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
                low_conf_mask = det[:, 4] < 0.51
                det[low_conf_mask, 4] = det[low_conf_mask, 4] + 0.2
                det[:, 4] = torch.clamp(det[:, 4], max=0.95)
    return pred


def parse_labelme_json_file(json_path):
    """Parse LabelMe JSON file and extract bounding boxes"""
    if not json_path or not os.path.exists(json_path):
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        boxes = []
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]
                label = shape.get('label', 'unknown')
                boxes.append({
                    'label': label,
                    'bbox': [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                })
        
        return boxes
    except Exception as e:
        print(f"Error parsing LabelMe JSON {json_path}: {e}")
        return []


def draw_comparison_boxes(image, gt_boxes, det_boxes):
    """Draw both GT (green) and Detection (blue) boxes on the same image"""
    if image is None:
        return None
    
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font_size = 24
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except:
        font = ImageFont.load_default(size=font_size)
    
    # Draw Ground Truth boxes in green
    for box in gt_boxes:
        bbox = box['bbox']
        label = f"GT: {box['label']}"
        draw.rectangle(bbox, outline='green', width=3)
        text_position = (bbox[0], max(0, bbox[1] - font_size - 5))
        text_bbox = draw.textbbox(text_position, label, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill='green')
        draw.text(text_position, label, fill='white', font=font)
    
    # Draw Detection boxes in blue
    for box in det_boxes:
        bbox = box['bbox']
        label = f"DET: {box.get('label', box.get('class', 'obj'))}"
        if 'confidence' in box:
            label += f" {box['confidence']:.2f}"
        draw.rectangle(bbox, outline='blue', width=3)
        text_position = (bbox[0], max(0, bbox[1] - font_size - 5))
        text_bbox = draw.textbbox(text_position, label, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill='blue')
        draw.text(text_position, label, fill='white', font=font)
    
    return img


def extract_bbox_pixels(image, boxes):
    """Extract pixel values from bounding boxes"""
    if image is None or len(boxes) == 0:
        return np.array([])
    
    img_array = np.array(image)
    all_pixels = []
    
    for box in boxes:
        if 'bbox' in box:
            bbox = box['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_array.shape[1], x2), min(img_array.shape[0], y2)
            
            bbox_pixels = img_array[y1:y2, x1:x2]
            if bbox_pixels.size > 0:
                if len(bbox_pixels.shape) == 3:
                    bbox_pixels_gray = np.mean(bbox_pixels, axis=2)
                else:
                    bbox_pixels_gray = bbox_pixels
                all_pixels.extend(bbox_pixels_gray.flatten())
    
    return np.array(all_pixels)


def create_individual_histogram(gt_pixels, det_pixels, image_name):
    """Create individual histogram for a single image (GT vs DET)"""
    plt.figure(figsize=(10, 5))
    
    if len(gt_pixels) > 0:
        hist_gt, bins_gt = np.histogram(gt_pixels, bins=50, range=(0, 255))
        bin_centers_gt = (bins_gt[:-1] + bins_gt[1:]) / 2
        plt.plot(bin_centers_gt, hist_gt, color='green', linewidth=2, 
                label=f'GT ({len(gt_pixels):,} pixels)', alpha=0.8)
    
    if len(det_pixels) > 0:
        hist_det, bins_det = np.histogram(det_pixels, bins=50, range=(0, 255))
        bin_centers_det = (bins_det[:-1] + bins_det[1:]) / 2
        plt.plot(bin_centers_det, hist_det, color='blue', linewidth=2, 
                label=f'DET ({len(det_pixels):,} pixels)', alpha=0.8)
    
    plt.title(f"Histogram: {image_name}", fontsize=12, fontweight='bold')
    plt.xlabel("Pixel Intensity", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(axis='both', alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    hist_img = Image.open(buf).copy()
    buf.close()
    plt.close()
    
    return hist_img


def create_aggregate_histogram_from_bins(gt_histogram, det_histogram):
    """Create aggregate histogram from pre-computed histogram bins (memory-efficient)"""
    plt.figure(figsize=(14, 7))
    
    bin_centers = np.arange(256)
    
    total_gt_pixels = int(gt_histogram.sum())
    total_det_pixels = int(det_histogram.sum())
    
    if total_gt_pixels > 0:
        plt.plot(bin_centers, gt_histogram, color='black', linewidth=2.5, 
                label=f'Ground Truth (Total pixels: {total_gt_pixels:,})', alpha=0.8)
    
    if total_det_pixels > 0:
        plt.plot(bin_centers, det_histogram, color='blue', linewidth=2.5, 
                label=f'Detection (Total pixels: {total_det_pixels:,})', alpha=0.8)
    
    plt.title("Aggregate Histogram: All Images Combined (GT vs DET)", fontsize=16, fontweight='bold')
    plt.xlabel("Pixel Intensity", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(axis='both', alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    hist_img = Image.open(buf).copy()
    buf.close()
    plt.close()
    
    return hist_img


def create_aggregate_histogram(all_gt_pixels, all_det_pixels):
    """Create aggregate histogram from all images"""
    plt.figure(figsize=(14, 7))
    
    if len(all_gt_pixels) > 0:
        hist_gt, bins_gt = np.histogram(all_gt_pixels, bins=50, range=(0, 255))
        bin_centers_gt = (bins_gt[:-1] + bins_gt[1:]) / 2
        plt.plot(bin_centers_gt, hist_gt, color='black', linewidth=2.5, 
                label=f'Ground Truth (Total pixels: {len(all_gt_pixels):,})', alpha=0.8)
    
    if len(all_det_pixels) > 0:
        hist_det, bins_det = np.histogram(all_det_pixels, bins=50, range=(0, 255))
        bin_centers_det = (bins_det[:-1] + bins_det[1:]) / 2
        plt.plot(bin_centers_det, hist_det, color='blue', linewidth=2.5, 
                label=f'Detection (Total pixels: {len(all_det_pixels):,})', alpha=0.8)
    
    plt.title("Aggregate Histogram: All Images Combined (GT vs DET)", fontsize=16, fontweight='bold')
    plt.xlabel("Pixel Intensity", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(axis='both', alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    hist_img = Image.open(buf).copy()
    buf.close()
    plt.close()
    
    return hist_img


def combine_image_and_histogram(comparison_img, histogram_img):
    """Combine comparison image and histogram side-by-side"""
    # Resize to same height
    target_height = 600
    
    # Resize comparison image
    comp_ratio = target_height / comparison_img.height
    comp_new_width = int(comparison_img.width * comp_ratio)
    comp_resized = comparison_img.resize((comp_new_width, target_height), Image.Resampling.LANCZOS)
    
    # Resize histogram
    hist_ratio = target_height / histogram_img.height
    hist_new_width = int(histogram_img.width * hist_ratio)
    hist_resized = histogram_img.resize((hist_new_width, target_height), Image.Resampling.LANCZOS)
    
    # Combine side-by-side
    total_width = comp_new_width + hist_new_width
    combined = Image.new('RGB', (total_width, target_height), 'white')
    combined.paste(comp_resized, (0, 0))
    combined.paste(hist_resized, (comp_new_width, 0))
    
    return combined


def process_single_image(image_path, json_path, model, class_names, conf_threshold, iou_threshold, actual_threshold):
    """Process a single image with its corresponding JSON"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    orig = image.copy()
    img = np.array(image)
    
    # Parse ground truth
    gt_boxes = parse_labelme_json_file(json_path)
    if not gt_boxes:
        return None
    
    # Preprocess for model
    img_for_model = letterbox(img, new_shape=(640, 640))[0]
    img_for_model = img_for_model.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_for_model).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    # Run detection
    with torch.no_grad():
        pred = model(img_tensor, augment=False, visualize=False)
    
    pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes=None, agnostic=False, max_det=1000)
    pred = adjust_confidence_ca_models(pred, "default", conf_threshold)
    
    # Process detections
    detections = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], (orig.height, orig.width)).round()
            for *xyxy, conf, cls in det:
                if conf >= actual_threshold:
                    cls_id = int(cls.item())
                    detections.append({
                        "class": class_names[cls_id],
                        "confidence": round(float(conf), 3),
                        "bbox": [round(float(x), 1) for x in xyxy],
                        "label": class_names[cls_id]
                    })
    
    # Extract pixels for histograms
    gt_pixels = extract_bbox_pixels(image, gt_boxes)
    det_pixels = extract_bbox_pixels(image, detections)
    
    # Create comparison image
    comparison_img = draw_comparison_boxes(image.copy(), gt_boxes, detections)
    
    return {
        'image_name': os.path.basename(image_path),
        'image': image,
        'gt_boxes': gt_boxes,
        'det_boxes': detections,
        'gt_pixels': gt_pixels,
        'det_pixels': det_pixels,
        'comparison_img': comparison_img,
        'gt_count': len(gt_boxes),
        'det_count': len(detections)
    }


def batch_process_folder(folder_path, model_choice, conf_threshold, iou_threshold, progress_callback=None):
    """Process all images in folder that have corresponding JSON files (memory-optimized with real-time progress)"""
    if not folder_path or not os.path.exists(folder_path):
        if progress_callback:
            yield "Please provide a valid folder path.", "", None, []
        return "Please provide a valid folder path.", "", None, []
    
    # Load model
    model_path = os.path.join("model", model_choice)
    device = "cpu"
    model = DetectMultiBackend(model_path, device=device)
    
    if "stomata" in model_choice.lower():
        class_names = ["st"]
    else:
        class_names = ["km", "non-km"]
    
    model.eval()
    actual_threshold = conf_threshold
    
    # Find all image-JSON pairs
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    pairs = []
    for img_file in folder.iterdir():
        if img_file.suffix.lower() in image_extensions:
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                pairs.append((str(img_file), str(json_file)))
    
    if not pairs:
        if progress_callback:
            yield "No image-JSON pairs found in the folder.", "", None, []
        return "No image-JSON pairs found in the folder.", "", None, []
    
    # Memory-optimized: Process in batches and accumulate histograms instead of all pixels
    total_images = len(pairs)
    successful_count = 0
    total_gt_boxes = 0
    total_det_boxes = 0
    
    # Use histograms instead of storing all pixels
    gt_histogram = np.zeros(256, dtype=np.int64)
    det_histogram = np.zeros(256, dtype=np.int64)
    
    # Store only essential info for gallery (limit to prevent OOM)
    MAX_GALLERY_IMAGES = 50
    gallery_results = []
    
    progress_text = f"Found {total_images} image-JSON pairs. Starting processing...\n\n"
    
    # Yield initial progress
    if progress_callback:
        yield "", progress_text, None, []
    
    for i, (img_path, json_path) in enumerate(pairs, 1):
        # Update progress
        current_progress = f"[{i}/{total_images}] Processing {os.path.basename(img_path)}... "
        
        try:
            result = process_single_image(img_path, json_path, model, class_names, 
                                        conf_threshold, iou_threshold, actual_threshold)
            if result:
                successful_count += 1
                total_gt_boxes += result['gt_count']
                total_det_boxes += result['det_count']
                
                # Accumulate histograms instead of storing all pixels
                if len(result['gt_pixels']) > 0:
                    hist, _ = np.histogram(result['gt_pixels'], bins=256, range=(0, 256))
                    gt_histogram += hist
                
                if len(result['det_pixels']) > 0:
                    hist, _ = np.histogram(result['det_pixels'], bins=256, range=(0, 256))
                    det_histogram += hist
                
                # Store only limited gallery images to save memory (with histogram)
                if len(gallery_results) < MAX_GALLERY_IMAGES:
                    # Create individual histogram for this image
                    individual_hist = create_individual_histogram(
                        result['gt_pixels'], 
                        result['det_pixels'], 
                        result['image_name']
                    )
                    
                    # Combine comparison image and histogram
                    combined_img = combine_image_and_histogram(
                        result['comparison_img'], 
                        individual_hist
                    )
                    
                    gallery_results.append({
                        'image_name': result['image_name'],
                        'comparison_img': combined_img,
                        'gt_count': result['gt_count'],
                        'det_count': result['det_count']
                    })
                
                # Clear pixel data from result to free memory
                del result['gt_pixels']
                del result['det_pixels']
                del result['image']
                
                current_progress += f"✓ GT={result['gt_count']}, DET={result['det_count']}\n"
            else:
                current_progress += f"⚠ No ground truth found\n"
        except Exception as e:
            current_progress += f"✗ Error: {str(e)}\n"
        
        progress_text += current_progress
        
        # Yield progress update in real-time
        if progress_callback:
            yield "", progress_text, None, gallery_results.copy()
        
        # Force garbage collection every 50 images
        if i % 50 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    progress_text += f"\n{'='*60}\n"
    progress_text += f"Processing Complete: {successful_count}/{total_images} images successfully processed\n"
    
    # Create aggregate histogram from accumulated histograms
    aggregate_hist = create_aggregate_histogram_from_bins(gt_histogram, det_histogram)
    
    # Summary statistics
    total_gt_pixels = int(gt_histogram.sum())
    total_det_pixels = int(det_histogram.sum())
    
    summary = f"""📊 Batch Processing Summary

✅ Successfully Processed: {successful_count} / {total_images} images
📦 Total Ground Truth Boxes: {total_gt_boxes:,}
🎯 Total Detection Boxes: {total_det_boxes:,}
🔍 Total GT Pixels Analyzed: {total_gt_pixels:,}
🔍 Total DET Pixels Analyzed: {total_det_pixels:,}

Average per image:
- GT boxes: {total_gt_boxes / successful_count:.1f}
- DET boxes: {total_det_boxes / successful_count:.1f}

💡 Gallery limited to first {min(MAX_GALLERY_IMAGES, successful_count)} images to conserve memory
    """
    
    # Final yield with all results
    if progress_callback:
        yield summary, progress_text, aggregate_hist, gallery_results
    else:
        return summary, progress_text, aggregate_hist, gallery_results


def create_individual_gallery(results):
    """Create gallery of individual comparisons with histograms (memory-optimized)"""
    if not results:
        return []
    
    gallery_images = []
    for result in results:
        if result.get('comparison_img'):
            # Add caption with stats
            caption = f"{result['image_name']}\nGT: {result['gt_count']} | DET: {result['det_count']}"
            gallery_images.append((result['comparison_img'], caption))
    
    return gallery_images


# Gradio Interface
def get_default_model():
    choices = get_model_choices()
    return choices[0] if choices else None


def process_batch(folder_path, model_choice, conf_threshold, iou_threshold, progress=gr.Progress()):
    """Main batch processing function with real-time updates"""
    # Use generator to get real-time updates
    generator = batch_process_folder(
        folder_path, model_choice, conf_threshold, iou_threshold, progress_callback=True
    )
    
    summary = ""
    progress_text = ""
    aggregate_hist = None
    gallery = []
    
    # Iterate through progress updates
    for result in generator:
        summary, progress_text, aggregate_hist, results = result
        gallery = create_individual_gallery(results)
        
        # Yield intermediate results for real-time display
        yield summary, progress_text, aggregate_hist, gallery
    
    # Final result is already yielded in the loop
    return summary, progress_text, aggregate_hist, gallery


with gr.Blocks(title="YOLOv5 Batch Processing") as demo:
    gr.Markdown("# YOLOv5 Batch Object Detection with Ground Truth Comparison")
    gr.Markdown("""
    This tool processes a folder of images with their corresponding LabelMe JSON annotations.
    - **Automatic Pairing**: Matches IMAGE_01.jpg with IMAGE_01.json
    - **Batch Processing**: Processes all matched pairs (memory-optimized for large datasets)
    - **Aggregate Analysis**: Combines all histograms for overall comparison
    - **Individual Results**: View first 50 images' comparisons (to conserve memory)
    - **💾 Memory Efficient**: Designed to handle 800+ images without running out of memory
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Configuration")
            folder_input = gr.Textbox(
                label="Folder Path", 
                placeholder="Enter the full path to your image folder (e.g., C:/path/to/images)",
                info="Folder should contain images (.jpg, .png) and their corresponding .json files"
            )
            
            model_dropdown = gr.Dropdown(
                choices=get_model_choices(), 
                value=get_default_model(), 
                label="Select Model"
            )
            
            conf_threshold = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=0.25, 
                step=0.05, 
                label="Confidence Threshold"
            )
            
            iou_threshold = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=0.45, 
                step=0.05, 
                label="IoU Threshold for NMS"
            )
            
            process_btn = gr.Button("🚀 Process Folder", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### Processing Summary")
            summary_output = gr.Textbox(
                label="📊 Summary Statistics", 
                lines=10, 
                max_lines=15,
                show_label=True
            )
            progress_output = gr.Textbox(
                label="📝 Processing Progress (n/N images processed)", 
                lines=10, 
                max_lines=20,
                show_label=True
            )
    
    with gr.Tabs():
        with gr.Tab("📊 Aggregate Histogram", id="aggregate"):
            gr.Markdown("""
            ### Combined Histogram of All Images
            This shows the pixel distribution across **all bounding boxes** from all processed images.
            - **Green Line**: All Ground Truth boxes combined
            - **Blue Line**: All Detection boxes combined
            """)
            aggregate_histogram = gr.Image(
                type="pil", 
                label="Aggregate Histogram (All Images Combined)", 
                height=500
            )
        
        with gr.Tab("🖼️ Individual Comparisons", id="individual"):
            gr.Markdown("""
            ### Individual Image Comparisons with Histograms
            Each entry shows:
            - **Left**: Image with overlayed Ground Truth (green) and Detection (blue) boxes
            - **Right**: Individual histogram comparing GT vs DET pixel distributions
            
            Click on any image to view it larger.
            
            **Note:** Gallery limited to first 50 images to conserve memory when processing large datasets.
            """)
            individual_gallery = gr.Gallery(
                label="Image Comparisons (Green=GT, Blue=DET)",
                show_label=True,
                columns=3,
                rows=2,
                height="auto",
                object_fit="contain"
            )
    
    # Event handler with streaming updates
    process_btn.click(
        fn=process_batch,
        inputs=[folder_input, model_dropdown, conf_threshold, iou_threshold],
        outputs=[summary_output, progress_output, aggregate_histogram, individual_gallery]
    )
    
    # Example usage
    gr.Markdown("""
    ---
    ### 📝 Usage Instructions:
    1. **Prepare your folder**: Ensure each image has a matching JSON file
       - Example: `image_01.jpg` + `image_01.json`
    2. **Enter folder path**: Paste the full path to your folder
    3. **Configure settings**: Adjust confidence and IoU thresholds as needed
    4. **Click Process**: Wait for batch processing to complete
    5. **View Results**: 
       - **Aggregate tab**: See overall histogram comparison
       - **Individual tab**: Browse each image's results
    
    ### 📂 Folder Structure Example:
    ```
    my_images/
    ├── image_01.jpg
    ├── image_01.json
    ├── image_02.jpg
    ├── image_02.json
    └── ...
    ```
    """)

if __name__ == "__main__":
    demo.launch()
