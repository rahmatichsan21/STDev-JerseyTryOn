# How to Use - Continuous Video Streaming with Capture

## ✅ What Changed

### Python Server (server.py)
- **Now streams video continuously** (not capture-on-demand)
- Video plays in real-time at ~30 FPS
- Can pause streaming when capture is requested
- Can resume streaming after capture

### Godot Client (Main.cs)
- **Receives continuous video stream** automatically
- Press **C** to capture and save current frame
- Press **R** to resume streaming after capture
- Saves images to `user://` directory with timestamp

## 🎮 Controls

| Key | Action |
|-----|--------|
| **C** | Capture current frame and save as PNG |
| **R** | Resume video streaming after capture |

## 📸 How It Works

### 1. Normal Operation (Video Playing)
```
Python → Streams frames continuously at ~30 FPS
   ↓
Godot → Displays frames in real-time
```

### 2. When You Press 'C' (Capture)
```
You press C
   ↓
Godot → Saves current frame to memory
   ↓
Godot → Sends "save_capture" command to Python
   ↓
Python → Pauses streaming
   ↓
Godot → Saves image to disk with timestamp
   ↓
Godot → Shows "Image saved" message
```

### 3. When You Press 'R' (Resume)
```
You press R
   ↓
Godot → Sends "resume" command to Python
   ↓
Python → Resumes streaming
   ↓
Video continues playing
```

## 💾 Saved Files

Images are saved to Godot's `user://` directory with filename format:
```
captured_YYYYMMDD_HHMMSS.png
```

Example: `captured_20251029_143052.png`

### Where are files saved?

On Windows, `user://` maps to:
```
%APPDATA%\Godot\app_userdata\[YourProjectName]\
```

Check Godot console for exact path - it prints the full path when saving!

## 🚀 Running the System

### 1. Start Python Server
```powershell
cd Scripts\Python
python server.py
```

You should see:
```
Server Python dimulai di ws://localhost:8765. Menunggu koneksi dari Godot...
```

### 2. Run Godot Scene
- Open your project in Godot
- Run the scene with Main.cs attached
- Video should start playing automatically

### 3. Capture Images
- Watch the video stream
- When you see a good frame, press **C**
- Check console for save confirmation
- Press **R** to continue streaming

## 🔍 Troubleshooting

### "Cannot capture: WebSocket not open"
**Solution**: Make sure Python server is running first

### "No video frame available to capture"
**Solution**: Wait a moment after connecting - first frame needs to arrive

### Video is frozen
**Solution**: Press **R** to resume streaming

### Can't find saved images
**Solution**: Check Godot console - it prints the full path where file was saved

## 📝 Technical Details

### Frame Rate
- Target: 30 FPS
- Actual: Depends on network and processing speed
- JPEG Quality: 60 (adjustable in server.py)

### Image Resolution
- Default: 640x480
- Adjustable in `server.py`:
  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  ```

### Save Format
- Format: PNG (lossless)
- Can be changed to JPEG in Main.cs:
  ```csharp
  lastCapturedImage.SaveJpg(filename);  // Instead of SavePng
  ```

## 🎯 Summary

- ✅ Video streams continuously
- ✅ Press C to capture
- ✅ Image saves automatically with timestamp
- ✅ Press R to resume
- ✅ Simple and straightforward!

Enjoy your webcam streaming and capture system! 📹✨
