# Jersey Try-On Integration Guide

## Overview
This integration allows Godot to capture images and send them to Python for jersey application processing.

## How It Works

1. **Godot captures an image** from the webcam
2. **Saves the image** to the user directory
3. **Sends processing request** to Python server via WebSocket
4. **Python processes the image**:
   - Detects the person's pose
   - Extracts the shirt region
   - Applies the selected jersey
   - Creates a smooth overlay
5. **Returns the processed image** path to Godot
6. **Godot can display** the result

## Setup

### 1. Start the Python Server

```powershell
cd Scripts\Python
python server.py
```

The server will start on `ws://localhost:8765`

### 2. Run Godot Application

Run the Godot project - it will automatically connect to the Python server.

## Usage Flow

1. **Press Capture Button** in Godot
   - Godot sends `save_capture` command
   - Python pauses streaming
   - Godot saves current frame

2. **Python Processes Image**
   - Receives `process_image` command with:
     - `image_path`: Path to captured image
     - `jersey`: Jersey name (e.g., "Arsenal Home.jpg")
     - `output_path`: Where to save result
   
3. **Processing Steps**:
   ```
   Load Image → Detect Pose → Extract Shirt Region
        ↓
   Detect Skin → Remove Skin from Shirt
        ↓
   Refine with GrabCut → Load Jersey
        ↓
   Fit Jersey to Shirt Shape → Create Smooth Overlay
        ↓
   Save Result → Send Response to Godot
   ```

4. **Result Ready**
   - Python sends `processing_complete` message
   - Godot receives the output file path
   - Can display or save the result

## File Locations

- **Captured Images**: `user://captured_YYYYMMDD_HHMMSS.png`
- **Processed Images**: `user://processed_YYYYMMDD_HHMMSS.png`
- **Jerseys**: `Assets/Jerseys/PremierLeague/Home/Processed/`

On Windows, `user://` typically maps to:
```
C:\Users\<YourName>\AppData\Roaming\Godot\app_userdata\<ProjectName>\
```

## Available Commands

### From Godot to Python:

1. **Save Capture**
```json
{
    "type": "save_capture"
}
```

2. **Process Image**
```json
{
    "type": "process_image",
    "image_path": "C:/path/to/image.png",
    "jersey": "Arsenal Home.jpg",
    "output_path": "C:/path/to/output.png"
}
```

3. **Resume Streaming**
```json
{
    "type": "resume"
}
```

### From Python to Godot:

1. **Capture Ready**
```json
{
    "type": "capture_ready",
    "message": "Frame ready to save"
}
```

2. **Processing Complete**
```json
{
    "type": "processing_complete",
    "success": true,
    "output_path": "C:/path/to/result.png"
}
```

3. **Error**
```json
{
    "type": "error",
    "message": "Error description"
}
```

## Customization

### Change Jersey Selection

In `Main.cs`, modify the `ProcessImageWithJersey` method:

```csharp
{"jersey", "Arsenal Home.jpg"},  // Change this to any jersey name
```

Available jerseys should be in:
```
Assets/Jerseys/PremierLeague/Home/Processed/
```

### Add Jersey Selection UI

You can create a dropdown/menu in Godot to let users select jerseys dynamically before capture.

## Troubleshooting

### Issue: No person detected
- **Solution**: Ensure the person is clearly visible in frame
- Check lighting conditions
- Person should be facing camera

### Issue: Jersey doesn't fit properly
- **Solution**: The system uses the detected shirt shape
- Make sure the shirt is clearly visible
- Try adjusting camera position

### Issue: Processing takes too long
- **Solution**: 
  - First run is slow (model loading)
  - Subsequent runs are faster
  - Consider resizing images before processing

### Issue: WebSocket connection fails
- **Solution**:
  - Ensure Python server is running
  - Check firewall settings
  - Verify port 8765 is not in use

## Next Steps

1. **Add UI for jersey selection**
2. **Display processing progress**
3. **Add preview before saving**
4. **Batch processing for multiple captures**
5. **Save comparison (before/after)**
