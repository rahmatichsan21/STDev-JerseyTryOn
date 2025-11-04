# Football Jersey Filter - ML System

## Quick Start

### 1. Install Dependencies
```bash
pip install opencv-python numpy scikit-learn scikit-image joblib websockets
```

### 2. Start Server
**Windows:** Double-click `START_SERVER.bat`  
**Manual:** `python jersey_filter_server.py`

### 3. Run Godot App
Open and run the Godot project - connects automatically to `ws://localhost:8765`

---

## System Overview

**Real-time jersey filter using traditional machine learning (Random Forest classifier)**

### Components:
- `jersey_filter_server.py` - WebSocket server with webcam + ML filter
- `shirt_detector_model.joblib` - Trained Random Forest model (91.9% accuracy)
- `shirt_detector_full.py` - ML training script (uses all 72 images in IcanDataset_NOBG)
- `apply_jersey.py` - Batch process images with jersey overlay
- `check_annotations.py` - Verify training data annotations

### How It Works:
1. Webcam captures frame
2. ML model detects shirt pixels (excludes hair, skin, arms using neck-level detection)
3. Brighton Home jersey overlays on shirt region only
4. Processed frame sent to Godot via WebSocket

### ML Model Details:
- **Type:** Random Forest (100 trees, max_depth=15)
- **Features:** 9D per pixel (RGB + HSV + YCrCb color spaces)
- **Training:** 288,000 pixels from 72 selfie images
- **Detection:** Neck-level boundary (finds face contours, excludes above neck+20px)
- **Performance:** ~30 FPS real-time, <1MB model size

---

## Training New Model

If you add more images to `DatasetImplementation/IcanDataset_NOBG/`:

```bash
cd DatasetImplementation
python shirt_detector_full.py
```

This will:
- Process all PNG images in IcanDataset_NOBG folder
- Use neck-level detection to exclude hair
- Train new Random Forest model
- Save to `shirt_detector_model.joblib`
- Test on 20% holdout set

---

## Batch Processing

Apply jersey to multiple images:

```bash
cd DatasetImplementation
python apply_jersey.py
```

Results saved to `final_jersey_output/` with comparison images.

---

## Troubleshooting

**Server won't start:** Check if port 8765 is in use: `netstat -an | findstr 8765`  
**Godot won't connect:** Restart both server and Godot app  
**No jersey appears:** Check webcam permissions and lighting  
**Model not found:** Run `shirt_detector_full.py` to train model first

---

## Dataset Structure

```
IcanDataset_NOBG/          72 selfie images (no background)
shirt_detector_model.joblib   Trained ML model
annotation_check/          Visualization of training annotations
final_jersey_output/       Batch processing results
```

---

## Technical Notes

- Uses **traditional ML only** (no GPU/deep learning required)
- Neck detection ensures hair is excluded from shirt region
- Jersey scales to fill shirt bounding box (no "apron" effect)
- Alpha blending for natural overlay
- Morphological operations for mask cleanup
