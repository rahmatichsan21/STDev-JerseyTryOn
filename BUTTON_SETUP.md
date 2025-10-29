# Button Setup Guide

## ✅ What Changed

Your system now:
1. **Video starts automatically** when connected (no button needed to start)
2. **Uses a button to capture** frames instead of keyboard
3. **Auto-resumes** video after saving (no need to press R)

## 🔘 Setting Up the Capture Button

### Option 1: If You Already Have a Button

The script will automatically find your button if it's named:
- `CaptureButton`
- `Button`
- `Panel/Button`
- `UI/CaptureButton`

**No extra setup needed!** Just run the scene and click your button to capture.

### Option 2: Creating a New Button

If you don't have a button yet:

1. **Open your scene** (`main.tscn` or `MainMenu.tscn`)

2. **Add a Button node**:
   - Right-click on your root Control node
   - Add Child Node → Button
   - Name it `CaptureButton`

3. **Position the button**:
   - Select the Button node
   - In the Inspector, set:
     - Position: Where you want it (e.g., bottom center)
     - Size: Make it big enough (e.g., 200x60)

4. **Customize the button text**:
   - In the Inspector → Text: `"📸 Capture"`
   - Or: `"Take Photo"`, `"Save Picture"`, etc.

5. **Style the button** (optional):
   - Theme → Font Size: Make it bigger
   - Theme → Colors → Font Color: Choose a color
   - Theme → Styles: Add background color

6. **Save the scene**

### Option 3: If Your Button Has a Different Name

If your button is named something else (like `TakePhotoButton` or `SnapButton`):

**Edit Main.cs line 27:**

```csharp
captureButton = GetNodeOrNull<Button>("YourButtonNameHere");
```

Replace `"YourButtonNameHere"` with your actual button's node path.

## 🎮 How It Works Now

### Automatic Video Start
```
Godot connects to Python
    ↓
Python starts streaming immediately
    ↓
Video appears on screen automatically
    ↓
No button press needed!
```

### Button Capture Flow
```
Video playing continuously
    ↓
User clicks Capture button (or presses C)
    ↓
Frame freezes briefly
    ↓
Image saved with timestamp
    ↓
Video resumes automatically
```

## 🎯 Controls

| Action | Method | Description |
|--------|--------|-------------|
| **Capture** | Click button OR press C | Saves current frame as PNG |
| **Resume** | Automatic OR press R | Video resumes after capture |

## 🏗️ Node Structure Example

```
Control (Main)
├── TextureRect (displays video)
└── CaptureButton (captures frame)
    └── Label (optional - button text)
```

## 🔍 Troubleshooting

### Button doesn't work
**Check console for:**
- `"Capture button connected!"` → Button found ✓
- `"Warning: CaptureButton not found"` → Button not found ✗

**Solutions:**
1. Make sure button is named `CaptureButton`
2. Make sure button is in the same scene as Main.cs
3. Check the node path is correct
4. Use C key as fallback for testing

### Video doesn't start automatically
**Solution:** Python server must be running **before** you start Godot scene

**Steps:**
1. Start Python server first: `python server.py`
2. Wait for: `"Server Python dimulai di ws://localhost:8765..."`
3. Then run Godot scene
4. Video should appear automatically

### "No video frame available to capture"
**Solution:** Wait 1-2 seconds after connecting before capturing

## 📋 Quick Setup Checklist

- [ ] Python server is running
- [ ] Godot scene has a Button node
- [ ] Button is named `CaptureButton` (or path is correct in code)
- [ ] TextureRect node exists to display video
- [ ] Run Godot scene
- [ ] Video appears automatically
- [ ] Click button to capture
- [ ] Check console for save confirmation

## 💡 Tips

1. **Better button visibility**: Add an icon or make it larger
2. **Countdown timer**: Add 3-2-1 countdown before capture
3. **Preview capture**: Show captured image briefly before saving
4. **Multiple saves**: Disable button during save to prevent double-clicks

## 🎨 Button Styling Example

In Godot Scene:

```
Button Properties:
├── Text: "📸 Take Photo"
├── Custom Fonts
│   └── Font Size: 24
├── Custom Colors
│   ├── Font Color: White
│   └── Font Pressed Color: Yellow
└── Custom Styles
    ├── Normal: Blue background
    ├── Hover: Light blue background
    └── Pressed: Dark blue background
```

---

**Your video now starts automatically and button capture is ready!** 🎥📸
