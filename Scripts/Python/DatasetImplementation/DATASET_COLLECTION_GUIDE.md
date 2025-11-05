# 📸 Panduan Lengkap: Membuat Dataset Jersey Try-On

## 🎯 Tujuan Dataset

Dataset yang baik harus mencakup **VARIASI MAKSIMAL** dari kondisi real-world yang akan dihadapi model saat runtime.

---

## 📊 Analisis Dataset Saat Ini

### Current Stats:
- **Total images:** ~69
- **Subjects:** 1-2 orang (homogen)
- **Poses:** Kebanyakan frontal, statis
- **Lighting:** Uniform (indoor, consistent)
- **Background:** Sudah removed (good!)
- **Shirt types:** Terbatas

### ⚠️ Kelemahan:
1. **Terlalu sedikit** (minimal butuh 200-500 images)
2. **Kurang variasi pose** (90% frontal)
3. **Kurang variasi subjek** (skin tone, body type)
4. **Kurang variasi pencahayaan**
5. **Kurang variasi jenis baju**

---

## ✅ Rekomendasi Dataset Ideal

### 🎯 Target Minimum:
- **Total images:** 300-500 images
- **Subjects:** 5-10 orang berbeda
- **Per subject:** 30-50 images dengan variasi
- **Total poses:** 15-20 kategori pose

---

## 📋 Kategori Pose Yang Harus Dikumpulkan

### 🔵 **Category 1: Basic Frontal Poses (30% - 150 images)**

Ini adalah pose dasar yang paling sering digunakan.

#### 1.1 Standing Straight - Frontal
**Priority: ★★★★★ (HIGHEST)**
- Berdiri tegak menghadap kamera
- Tangan di samping badan
- Kepala lurus
- **Variations:**
  - Tangan santai
  - Tangan di saku (sebagian)
  - Slight head tilt (kiri/kanan)
- **Jumlah:** 20-30 images per subject

**Contoh:**
```
👤 Frontal, tangan di samping
👤 Frontal, tangan di saku depan
👤 Frontal, slight smile
```

#### 1.2 Standing with Arms Variations
**Priority: ★★★★☆**
- **Tangan di pinggang** (both hands)
- **Tangan silang di dada** (crossed arms)
- **Satu tangan di pinggang**, satu di samping
- **Thumbs up pose**
- **Peace sign pose**
- **Jumlah:** 15-20 images per subject

**Contoh:**
```
👤 Hands on hips
👤 Arms crossed
👤 One hand on hip
```

---

### 🟢 **Category 2: Rotation & Angle Variations (25% - 125 images)**

Model harus bisa detect shirt dari berbagai sudut.

#### 2.1 Slight Rotation (15-30 degrees)
**Priority: ★★★★★**
- **Quarter turn kiri** (15-30°)
- **Quarter turn kanan** (15-30°)
- Masih terlihat chest area
- **Jumlah:** 15-20 images per subject

**Contoh:**
```
   👤  (kamera)   Slightly turned left
👤      (kamera)   Frontal
      👤(kamera)   Slightly turned right
```

#### 2.2 Side Profile (45-60 degrees)
**Priority: ★★★☆☆**
- **Half profile kiri**
- **Half profile kanan**
- Shirt side visible
- **Jumlah:** 10-15 images per subject

#### 2.3 Full Side (90 degrees)
**Priority: ★★☆☆☆**
- **Full side kiri**
- **Full side kanan**
- Hanya untuk edge cases
- **Jumlah:** 5-10 images per subject

---

### 🟡 **Category 3: Action Poses (20% - 100 images)**

Pose dinamis yang mencerminkan real-world usage.

#### 3.1 Arms Raised/Extended
**Priority: ★★★★☆**
- **Satu tangan diangkat** (waving)
- **Dua tangan diangkat**
- **Tangan terentang ke samping**
- **Pointing gesture**
- **Jumlah:** 15-20 images per subject

**Why important:** Shirt akan stretch dan deform!

#### 3.2 Bending/Leaning
**Priority: ★★★☆☆**
- **Slight lean forward**
- **Slight lean backward**
- **Lean to side**
- **Jumlah:** 10-15 images per subject

**Why important:** Body angle changes shirt visibility.

#### 3.3 Seated Poses
**Priority: ★★★☆☆**
- **Duduk tegak**
- **Duduk santai**
- **Duduk dengan tangan di lutut**
- **Jumlah:** 10-15 images per subject

**Why important:** Shirt gathers/bunches differently.

---

### 🟣 **Category 4: Distance Variations (15% - 75 images)**

#### 4.1 Different Distances from Camera
**Priority: ★★★★☆**
- **Close up** (shoulders & head only)
- **Medium** (waist up) - **MOST COMMON**
- **Full body** (entire person)
- **Far** (person smaller in frame)
- **Jumlah:** 15-20 images per subject

**Why important:** Shirt size in frame varies!

---

### 🔴 **Category 5: Edge Cases & Challenging Scenarios (10% - 50 images)**

#### 5.1 Partial Occlusion
**Priority: ★★★☆☆**
- **Hand partially covering shirt**
- **Holding object** (phone, bag, water bottle)
- **Wearing jacket** (open, showing shirt underneath)
- **Scarf/accessories**
- **Jumlah:** 10-15 images per subject

#### 5.2 Extreme Poses
**Priority: ★★☆☆☆**
- **Arms fully crossed**
- **Hugging self**
- **Stretching**
- **Looking down**
- **Looking up**
- **Jumlah:** 5-10 images per subject

---

## 👥 Subject Diversity Requirements

### Minimum 5-10 Subjects dengan Variasi:

#### 1. **Skin Tone Diversity** ⭐⭐⭐⭐⭐
- Fair skin
- Medium skin
- Tan skin
- Dark skin

**Why:** Color-based detection sangat sensitif terhadap skin tone!

#### 2. **Body Type Diversity** ⭐⭐⭐⭐
- Slim/athletic
- Average
- Plus size

**Why:** Shirt shape berbeda!

#### 3. **Gender Diversity** ⭐⭐⭐
- Male
- Female
- Unisex fits

**Why:** Body proportions berbeda.

#### 4. **Age Diversity** ⭐⭐☆
- Young adults (18-25)
- Adults (26-40)
- Middle age (40+)

---

## 👕 Shirt Type Variations

### Minimum Shirt Types per Subject:

#### 1. **Solid Colors** ⭐⭐⭐⭐⭐ (ESSENTIAL)
- **White shirt** (paling challenging!)
- **Black shirt**
- **Red shirt**
- **Blue shirt**
- **Green shirt**
- **Yellow shirt**

**Why:** Model harus bisa detect ANY color!

#### 2. **Patterns** ⭐⭐⭐⭐
- **Striped shirt** (horizontal/vertical)
- **Checkered/plaid**
- **Graphic tee** (dengan print/logo)
- **Solid with logo**

#### 3. **Shirt Types** ⭐⭐⭐
- **T-shirt** (short sleeve) - PRIMARY
- **Polo shirt**
- **Long sleeve shirt**
- **V-neck**
- **Tank top**

---

## 💡 Lighting Variations

### Indoor Lighting (60%):
- **Bright daylight** (near window)
- **Normal room lighting**
- **Dim lighting**
- **Warm lighting** (yellow tone)
- **Cool lighting** (white/blue tone)

### Outdoor Lighting (40%):
- **Direct sunlight**
- **Cloudy day**
- **Shade**
- **Golden hour** (sunset)

**Why important:** Skin & shirt color detection depends on lighting!

---

## 📐 Camera Setup Guidelines

### Distance from Camera:
- **Recommended:** 1.5 - 2.5 meters
- **Range:** 1 - 4 meters

### Camera Height:
- **Eye level** (most natural)
- **Chest level** (good for shirt focus)
- **Slight above** (slight downward angle)

### Resolution:
- **Minimum:** 640x480
- **Recommended:** 1280x720 or higher
- **Will be resized**, tapi start dengan quality bagus

### Background:
- **White/neutral background** (BEST)
- **Plain wall**
- **Clean background** (easy to remove)
- **Avoid:** Cluttered, similar color to shirt

---

## 📝 Annotation Guidelines

Setiap foto harus di-annotate dengan **bounding box** untuk shirt region.

### Bounding Box Rules:

#### ✅ DO:
- **Include entire visible shirt area**
- **Include from neck/collar to bottom hem**
- **Include both sleeves** (if visible)
- **Tight box** (minimal extra space)

#### ❌ DON'T:
- Include face/hair
- Include too much background
- Cut off sleeves
- Include pants/lower body

### Annotation Format (CSV):
```csv
filename,x_min,y_min,x_max,y_max
subject1_pose1.png,100,150,500,450
subject1_pose2.png,120,160,520,470
```

---

## 🎬 Step-by-Step Collection Process

### Phase 1: Setup (30 minutes)
1. ✅ **Prepare space:**
   - Clear area with good lighting
   - White/neutral backdrop
   - Camera on tripod (if possible)

2. ✅ **Prepare subjects:**
   - 5-10 volunteers
   - Different body types, skin tones
   - Comfortable clothing

3. ✅ **Prepare shirts:**
   - Multiple colors (minimum 5 colors)
   - Different patterns
   - Clean, wrinkle-free

### Phase 2: Photoshoot (2-3 hours per subject)

**For EACH subject:**

1. **Session 1 - Basic Frontal (20 min)**
   - 30 photos: Standing straight variations
   - Change lighting 3-4 times

2. **Session 2 - Rotations (15 min)**
   - 20 photos: Left/right rotations (15°, 30°, 45°)
   - Front, quarter, half profile

3. **Session 3 - Action Poses (20 min)**
   - 20 photos: Arms raised, crossed, pointing
   - Dynamic movements

4. **Session 4 - Seated (15 min)**
   - 15 photos: Various sitting positions

5. **Session 5 - Edge Cases (10 min)**
   - 10 photos: Occlusions, extreme poses

6. **Change Shirt & Repeat** (if time permits)
   - At least 2-3 different shirt colors per subject

**Total per subject:** ~100 photos × 5 subjects = **500 photos**

### Phase 3: Background Removal (2-3 hours)
- Use tools: rembg, remove.bg, Photoshop
- Batch process jika memungkinkan
- Save as PNG dengan transparency

### Phase 4: Annotation (3-4 hours)
- Manual annotation (bisa pakai tools)
- **Tools recommendation:**
  - LabelImg (free, easy)
  - CVAT (online)
  - VGG Image Annotator
  
- Save annotations as CSV

### Phase 5: Validation (1 hour)
- Check annotations
- Run check script
- Fix errors

---

## 🛠️ Recommended Tools

### Photo Capture:
- **Smartphone camera** (modern phones are great!)
- **Webcam** (jika consistent setup)
- **DSLR/mirrorless** (if available)

### Background Removal:
```bash
pip install rembg
rembg i input.jpg output.png
```
Or use: remove.bg, Photoshop, GIMP

### Annotation:
- **LabelImg:** https://github.com/heartexlabs/labelImg
- **CVAT:** https://cvat.org
- **VGG Image Annotator:** https://www.robots.ox.ac.uk/~vgg/software/via/

---

## 📊 Dataset Quality Checklist

### Before Collection:
- [ ] 5+ subjects recruited
- [ ] 5+ shirt colors prepared
- [ ] Good lighting setup
- [ ] Neutral background ready
- [ ] Camera tested

### During Collection:
- [ ] Each subject: 80-100 photos
- [ ] Multiple poses covered
- [ ] Multiple shirt colors per subject
- [ ] Lighting variations captured
- [ ] Distance variations included

### After Collection:
- [ ] Total images: 300-500+
- [ ] Background removed (all images)
- [ ] Annotations completed
- [ ] Annotations validated
- [ ] Dataset split: 80% train, 20% test

---

## 🎯 Priority Order (If Time/Resources Limited)

### MUST HAVE (Critical):
1. ✅ Basic frontal poses (100 images)
2. ✅ Multiple subjects (3+ people)
3. ✅ Multiple shirt colors (5+ colors)
4. ✅ Rotation variations (±30°)

### SHOULD HAVE (Important):
5. ✅ Action poses (arms raised, crossed)
6. ✅ Distance variations
7. ✅ Lighting variations

### NICE TO HAVE (Enhancement):
8. ✅ Seated poses
9. ✅ Edge cases
10. ✅ Extreme poses

---

## 📈 Expected Improvement

### Current Dataset (~69 images):
- **Accuracy:** ~70-75%
- **Robustness:** Low (fails on new poses/lighting)
- **Generalization:** Poor

### Improved Dataset (300-500 images with diversity):
- **Accuracy:** ~85-90%
- **Robustness:** Medium-High
- **Generalization:** Good
- **Real-world performance:** Much better!

---

## 🚀 Quick Start Checklist

### Minimum Viable Dataset (1 day effort):

**Target: 200 images**

**Subjects:** 4 people
**Per subject:** 50 images

#### Per Subject Session:
- [ ] **Frontal poses** (20 images)
  - Standing straight: 5 images
  - Arms variations: 15 images

- [ ] **Rotations** (15 images)
  - Left 30°: 5 images
  - Frontal: 5 images
  - Right 30°: 5 images

- [ ] **Action poses** (10 images)
  - Arms raised: 5 images
  - Pointing/gestures: 5 images

- [ ] **Distance/other** (5 images)
  - Close, medium, far variations

**Total:** 4 subjects × 50 images = **200 images**

---

## 💻 Helper Script: Batch File Renaming

```python
# rename_dataset.py
import os
from pathlib import Path

def rename_dataset_images(dataset_dir, subject_name, session_number):
    """
    Rename images to standardized format:
    subject01_session1_pose001.png
    """
    images = sorted(Path(dataset_dir).glob("*.png"))
    
    for idx, img_path in enumerate(images, start=1):
        new_name = f"{subject_name}_session{session_number}_pose{idx:03d}.png"
        new_path = img_path.parent / new_name
        img_path.rename(new_path)
        print(f"Renamed: {img_path.name} -> {new_name}")

# Usage:
# rename_dataset_images("photos/", "subject01", 1)
```

---

## 📚 Additional Resources

### Dataset Inspirations:
- **DeepFashion Dataset** (pose references)
- **COCO Person Keypoints** (pose variety)
- **Fashionpedia** (clothing annotations)

### Annotation Best Practices:
- Consistent bounding box size
- Include collar/neckline
- Exclude face entirely
- Include visible sleeves

### Quality Control:
- Review 10% random sample
- Check annotation alignment
- Verify background removal quality

---

## ✅ Final Tips

### DO's:
- ✅ **Diversity is key** - different people, poses, colors
- ✅ **Consistency** - same annotation standards
- ✅ **Quality > Quantity** - 200 good images > 500 bad ones
- ✅ **Document everything** - note taking during photoshoot
- ✅ **Incremental training** - add data, retrain, test

### DON'Ts:
- ❌ **Don't rush** - bad annotations waste training time
- ❌ **Don't skip background removal** - affects training quality
- ❌ **Don't use only one subject** - model won't generalize
- ❌ **Don't use only one color shirt** - model overfits
- ❌ **Don't annotate with face included** - confuses model

---

## 🎉 Success Metrics

After collecting and training with new dataset:

### Test These Scenarios:
1. ✅ New person (not in training set)
2. ✅ New shirt color
3. ✅ New pose
4. ✅ Different lighting
5. ✅ Rotation (±45°)

### Expected Results:
- **Current:** 2-3 out of 5 work well
- **After improvement:** 4-5 out of 5 work well

---

**Good luck with your dataset collection! 🚀**

---

*Created for STDev-JerseyTryOn Project*  
*Last updated: November 2025*
