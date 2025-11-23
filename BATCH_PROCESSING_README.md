# YOLOv5 Batch Processing Application

## Overview

`app_batch.py` is a specialized version of the YOLOv5 detection application designed for **batch processing** of multiple images with their corresponding LabelMe JSON annotations.

## Key Features

### 1. **Automatic Image-JSON Pairing** 🔗
- Automatically finds and matches images with their JSON files
- Supports common image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Example pairing:
  ```
  IMAGE_01.jpg  ←→  IMAGE_01.json
  IMAGE_02.jpg  ←→  IMAGE_02.json
  ```

### 2. **Batch Detection Processing** ⚡
- Processes entire folders of images in one go
- Runs YOLOv5 detection on each image
- Compares detections against ground truth annotations
- Provides progress feedback for each image

### 3. **Aggregate Histogram Analysis** 📊
- Combines pixel data from **all** bounding boxes across **all** images
- Creates a single comprehensive histogram showing:
  - **Green line**: All ground truth boxes combined
  - **Blue line**: All detection boxes combined
- Useful for:
  - Overall model performance assessment
  - Dataset-wide pixel distribution analysis
  - Publication-quality aggregate statistics

### 4. **Individual Image Comparisons** 🖼️
- Browse each processed image separately
- Visual overlay showing:
  - **Green boxes**: Ground Truth annotations
  - **Blue boxes**: Model detections with confidence scores
- Gallery view for easy navigation
- Click to enlarge any image

## How It Works

### Processing Flow:

```
┌─────────────────────────────────────────────────────────┐
│ 1. User selects folder containing images + JSON files  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. System scans folder for image-JSON pairs            │
│    - Matches: IMAGE_01.jpg + IMAGE_01.json             │
│    - Skips images without JSON                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. For each pair:                                       │
│    a. Load image and parse LabelMe JSON                 │
│    b. Run YOLOv5 detection                              │
│    c. Extract pixels from GT and DET bounding boxes     │
│    d. Create comparison visualization                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Aggregate all results:                               │
│    - Combine all GT pixels → aggregate histogram       │
│    - Combine all DET pixels → aggregate histogram      │
│    - Create gallery of individual comparisons           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Display results in two tabs:                        │
│    - Aggregate Histogram Tab                            │
│    - Individual Comparisons Tab (Gallery)               │
└─────────────────────────────────────────────────────────┘
```

## User Interface

### Input Panel (Left Side):
- **Folder Path**: Text input for folder location
- **Select Model**: Dropdown to choose YOLOv5 model
- **Confidence Threshold**: Slider (0.0 - 1.0)
- **IoU Threshold**: Slider for NMS (0.0 - 1.0)
- **Process Folder Button**: Start batch processing

### Output Panel (Right Side):
- **📊 Summary Statistics**: Total counts, averages, and metrics
- **📝 Processing Progress**: Real-time progress with `[n/N]` counter showing how many images have been processed

### Results Tabs:

#### Tab 1: Aggregate Histogram 📊
- Single combined histogram
- Shows overall GT vs DET pixel distribution
- Includes total pixel counts
- Best for dataset-level analysis

#### Tab 2: Individual Comparisons 🖼️
- Gallery of all processed images
- Each image shows GT (green) + DET (blue) boxes
- Captions show GT and DET counts
- Click to view full size

## Usage Example

### Step 1: Prepare Your Folder
```
my_dataset/
├── stomata_001.jpg
├── stomata_001.json
├── stomata_002.jpg
├── stomata_002.json
├── stomata_003.jpg
├── stomata_003.json
└── ...
```

### Step 2: Launch Application
```bash
python app_batch.py
```

### Step 3: Configure Settings
1. Enter folder path: `C:/Users/YourName/my_dataset`
2. Select model: `yolov5n6-km-ca/weights/best.pt`
3. Set confidence threshold: `0.25`
4. Set IoU threshold: `0.45`

### Step 4: Process
Click **"🚀 Process Folder"** and wait for completion

### Step 5: View Results
- Check **Aggregate Histogram** tab for overall analysis
- Browse **Individual Comparisons** tab for per-image results

## Output Information

### Summary Statistics Shows:
- Total images processed
- Total ground truth boxes across all images
- Total detection boxes across all images
- Total pixels analyzed from GT boxes
- Total pixels analyzed from DET boxes

### Processing Log Shows:
- **Progress counter**: `[n/N]` format showing current/total
- Per-image status with checkmarks
- GT and DET counts for each image
- Error messages for failed images
- Final completion summary

Example output:
```
Found 10 image-JSON pairs. Starting processing...

[1/10] Processing image_01.jpg... ✓ GT=5, DET=4
[2/10] Processing image_02.jpg... ✓ GT=3, DET=3
[3/10] Processing image_03.jpg... ⚠ No ground truth found
[4/10] Processing image_04.jpg... ✓ GT=7, DET=6
[5/10] Processing image_05.jpg... ✗ Error: Invalid image format
[6/10] Processing image_06.jpg... ✓ GT=2, DET=2
[7/10] Processing image_07.jpg... ✓ GT=4, DET=5
[8/10] Processing image_08.jpg... ✓ GT=6, DET=5
[9/10] Processing image_09.jpg... ✓ GT=3, DET=3
[10/10] Processing image_10.jpg... ✓ GT=8, DET=7

============================================================
Processing Complete: 8/10 images successfully processed
```

## Advantages Over Single Image Processing

| Feature | Single Image App | Batch App |
|---------|-----------------|-----------|
| Processing Speed | Manual, one at a time | Automatic, entire folder |
| Dataset Analysis | Individual only | Aggregate + Individual |
| Time Required | High (manual clicks) | Low (one click) |
| Histogram Scope | Single image | All images combined |
| Use Case | Quick testing | Dataset validation |
| Best For | Development | Production/Research |

## Technical Details

### Supported Image Formats:
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tiff`)

### LabelMe JSON Requirements:
- Must be named identically to image (except extension)
- Must contain `shapes` array with rectangles
- Format: `shape_type: "rectangle"`

### Memory Considerations:
- Pixels are accumulated in memory for aggregate histogram
- For very large datasets (1000+ images), may require significant RAM
- Comparison images are stored temporarily for gallery

### Performance:
- Processes images sequentially (not parallel)
- Speed depends on:
  - Model size
  - Image resolution
  - Number of objects per image
  - CPU performance

## Common Use Cases

### 1. **Dataset Validation**
Verify that your entire dataset is properly annotated and detectable

### 2. **Model Performance Analysis**
Assess model performance across all test images at once

### 3. **Publication Figures**
Generate aggregate histograms for research papers

### 4. **Quality Assurance**
Quickly identify problematic images or annotations

### 5. **Batch Inference**
Run detection on multiple images without manual intervention

## Comparison with `app.py`

| Feature | app.py | app_batch.py |
|---------|--------|--------------|
| Input | Single image | Folder of images |
| Processing | Interactive | Batch automated |
| Histograms | Individual image | Aggregate + Individual |
| Layer Analysis | ✓ Available | ✗ Not included |
| Model Structure | ✓ Available | ✗ Not included |
| Layer Code Viewer | ✓ Available | ✗ Not included |
| Best Use | Development/Testing | Production/Validation |

## Tips for Best Results

1. **Organize Your Data**: Keep images and JSON files in the same folder
2. **Use Consistent Naming**: Ensure JSON files match image names exactly
3. **Check JSON Format**: Verify all JSON files are valid LabelMe format
4. **Monitor Progress**: Watch the processing log for any errors
5. **Start Small**: Test with a few images first before processing large datasets
6. **Adjust Thresholds**: Fine-tune confidence/IoU based on your needs

## Error Handling

The application handles:
- Missing JSON files (skips those images)
- Invalid JSON format (logs error, continues)
- Image loading errors (logs error, continues)
- Empty detections (continues processing)

Errors are shown in the Processing Log with details.

## Running the Application

### Development Mode:
```bash
python app_batch.py
```

### With Custom Port:
```python
# In app_batch.py, modify last line:
demo.launch(server_port=7861)
```

### Share Mode (Public URL):
```python
# In app_batch.py, modify last line:
demo.launch(share=True)
```

## Future Enhancements (Potential)

- [ ] Export results to CSV/Excel
- [ ] Parallel processing for faster batch execution
- [ ] Progress bar with percentage completion
- [ ] Filtering by confidence/class
- [ ] Export aggregate histogram data (.npy)
- [ ] Comparison metrics (IoU, Precision, Recall)
- [ ] Confusion matrix for batch results

## License

Same as parent YOLOv5 project.

## Support

For issues or questions, refer to the main project documentation or create an issue in the repository.
