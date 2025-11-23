# Quick Start Guide: app.py vs app_batch.py

## Which App Should You Use?

### Use `app.py` when:
- ✅ Testing individual images
- ✅ Need detailed layer analysis
- ✅ Want to see model architecture
- ✅ Developing/debugging
- ✅ Need layer code inspection
- ✅ Interactive exploration

### Use `app_batch.py` when:
- ✅ Processing entire datasets
- ✅ Need aggregate statistics
- ✅ Validating multiple annotations
- ✅ Creating publication figures
- ✅ Batch inference required
- ✅ Quality assurance testing

---

## Running Both Apps

### Run Single Image App:
```bash
python app.py
```
Opens at: http://localhost:7860

### Run Batch Processing App:
```bash
python app_batch.py
```
Opens at: http://localhost:7860

**Note**: Run them separately (not at the same time on same port)

---

## Feature Comparison Table

| Feature | app.py | app_batch.py |
|---------|:------:|:------------:|
| **Input** |
| Single Image Upload | ✅ | ❌ |
| Folder Input | ❌ | ✅ |
| LabelMe JSON Upload | ✅ | ✅ Auto-detect |
| **Processing** |
| Object Detection | ✅ | ✅ |
| Ground Truth Comparison | ✅ | ✅ |
| Batch Processing | ❌ | ✅ |
| **Analysis** |
| Individual Image Histogram | ✅ | ✅ |
| Aggregate Histogram | ❌ | ✅ |
| Layer Analysis | ✅ | ❌ |
| Model Architecture Diagram | ✅ | ❌ |
| Layer Code Viewer | ✅ | ❌ |
| Inference Metrics | ✅ | ❌ |
| **Outputs** |
| Detection Summary | ✅ | ✅ |
| Individual Comparison | ✅ | ✅ Gallery |
| Combined GT+DET Image | ✅ | ✅ |
| Box Histograms | ✅ | ✅ |
| Background Histograms | ✅ | ❌ |
| Overlayed Line Histogram | ✅ | ✅ |
| Downloadable NPY Files | ✅ | ❌ |
| Processing Log | ❌ | ✅ |
| Batch Statistics | ❌ | ✅ |
| **Gallery View** | ❌ | ✅ |

---

## Workflow Examples

### Research Workflow: Single Image Deep Dive
```
1. Open app.py
2. Upload test image + JSON
3. Analyze layer outputs
4. Check model architecture
5. Review code implementation
6. Adjust thresholds
7. Export histogram data
```

### Production Workflow: Dataset Validation
```
1. Open app_batch.py
2. Enter dataset folder path
3. Configure thresholds
4. Process entire folder
5. Review aggregate histogram
6. Browse individual results
7. Identify problematic images
```

---

## Typical Use Cases

### app.py Use Cases:
1. **Development**: Testing new models on sample images
2. **Debugging**: Understanding layer-by-layer processing
3. **Model Selection**: Comparing different architectures
4. **Threshold Tuning**: Finding optimal confidence/IoU values
5. **Educational**: Learning how YOLOv5 works internally

### app_batch.py Use Cases:
1. **Dataset Validation**: Verify all annotations are correct
2. **Performance Evaluation**: Test model on entire test set
3. **Quality Control**: Find mislabeled or problematic images
4. **Batch Inference**: Process multiple images for deployment
5. **Research**: Generate aggregate statistics for papers

---

## Quick Decision Tree

```
Need to process multiple images?
│
├─ NO → Use app.py
│       Need layer analysis or model details?
│       ├─ YES → Use app.py ✓
│       └─ NO  → Still use app.py (single image processing)
│
└─ YES → Use app_batch.py
         Need aggregate statistics across all images?
         ├─ YES → Use app_batch.py ✓
         └─ NO  → Still use app_batch.py (easier than manual)
```

---

## Installation & Setup

Both apps use the same dependencies:

```bash
# Install requirements (same for both apps)
pip install -r requirements.txt

# Ensure model folder exists
ls model/  # Should show your trained models

# Ensure yolov5 folder exists
ls yolov5/  # Should show YOLOv5 source code
```

---

## Tips & Tricks

### For app.py:
- Use "Show Code" button to understand layer implementations
- Download .npy files for custom analysis in Python
- Adjust magnifier settings for density calculations
- Use layer analysis to debug model issues

### For app_batch.py:
- Start with small folder (5-10 images) for testing
- Monitor processing log for errors
- Use aggregate histogram for overall quality assessment
- Click gallery images to view full resolution

---

## Future Integration

Potential future version could combine both:
- **app_combined.py**: Single app with tab for single/batch processing
- Switch between modes dynamically
- Reuse model loading across modes
- Unified settings and configuration

---

## Support & Documentation

- **Single Image App**: See main README.md and GROUND_TRUTH_COMPARISON_FEATURES.md
- **Batch Processing App**: See BATCH_PROCESSING_README.md
- **General Help**: Check YOLOv5 documentation

---

**Remember**: Both apps complement each other!
- Use app.py for deep analysis
- Use app_batch.py for broad coverage
