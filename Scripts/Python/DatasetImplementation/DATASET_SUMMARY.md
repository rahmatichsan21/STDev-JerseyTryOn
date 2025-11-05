# 🎯 RINGKASAN CEPAT: Pose untuk Dataset Jersey Try-On

## ⚡ TL;DR - Yang HARUS Dikumpulkan

### PRIORITAS TERTINGGI (60% dataset):

#### 1. **FRONTAL - STANDING** ⭐⭐⭐⭐⭐
- Tangan di samping
- Tangan di pinggang
- Tangan silang di dada
- **Target: 40 foto per orang**

#### 2. **ROTASI** ⭐⭐⭐⭐⭐  
- Putar 30° ke kiri
- Frontal
- Putar 30° ke kanan
- **Target: 20 foto per orang**

#### 3. **TANGAN AKTIF** ⭐⭐⭐⭐
- Satu tangan diangkat
- Dua tangan diangkat
- Gesture (pointing, peace sign)
- **Target: 15 foto per orang**

#### 4. **DUDUK** ⭐⭐⭐
- Duduk tegak
- Duduk santai
- **Target: 10 foto per orang**

---

## 📊 Target Dataset Minimum

### **200 Images (1 hari effort)**
- **4 orang** × **50 foto** = 200 images

### **400 Images (2 hari effort) - RECOMMENDED**
- **5 orang** × **80 foto** = 400 images

---

## 👥 Diversity Yang Dibutuhkan

### ✅ HARUS ADA:
1. **Minimal 3-5 orang berbeda**
   - Beda skin tone (terang, medium, gelap)
   - Beda body type (kurus, average, gemuk)
   - Beda gender (kalau memungkinkan)

2. **Minimal 5 warna baju berbeda per orang**
   - ⚪ Putih (CRITICAL!)
   - ⚫ Hitam
   - 🔴 Merah/Biru/Hijau (pilih minimal 3)

3. **Variasi pencahayaan**
   - Terang (dekat jendela)
   - Normal (lampu ruangan)
   - Redup (jauh dari jendela)

---

## 📸 Checklist Foto Per Orang

### Session 1: FRONTAL (20 menit)
- [ ] Standing straight - tangan samping (10 foto)
- [ ] Hands on hips (5 foto)
- [ ] Arms crossed (5 foto)
- [ ] Peace/thumbs up (5 foto)

### Session 2: ROTASI (15 menit)
- [ ] Turn left 30° (5 foto)
- [ ] Frontal repeat (5 foto)
- [ ] Turn right 30° (5 foto)
- [ ] Side 45° (5 foto)

### Session 3: AKSI (15 menit)
- [ ] One arm raised (5 foto)
- [ ] Both arms raised (5 foto)
- [ ] Pointing/gestures (5 foto)

### Session 4: DUDUK (10 menit)
- [ ] Seated straight (5 foto)
- [ ] Seated relaxed (5 foto)

**Total: 80 foto per orang (~60 menit)**

---

## 🎨 Urutan Pengambilan Foto

### Per Orang:
1. **Sesi 1:** Baju putih (80 foto)
2. **Break** (5 menit, ganti baju)
3. **Sesi 2:** Baju hitam (40 foto - pose penting saja)
4. **Break** (5 menit, ganti baju)  
5. **Sesi 3:** Baju warna (40 foto - pose penting saja)

**Total: 160 foto per orang (2 jam)**

### 5 Orang:
**Total: 800 foto (tapi pilih 400 terbaik)**

---

## ⚠️ KESALAHAN YANG HARUS DIHINDARI

### ❌ JANGAN:
1. **Hanya 1-2 orang** → Model tidak generalize
2. **Hanya 1-2 warna baju** → Model overfit ke warna tertentu
3. **Hanya pose frontal statis** → Gagal di pose lain
4. **Background tidak dihapus** → Training jadi buruk
5. **Annotasi asal-asalan** → Waste training time

### ✅ LAKUKAN:
1. **5+ orang** dengan diversity (skin, body type)
2. **5+ warna** baju per orang (putih, hitam, + 3 warna)
3. **15+ pose** berbeda per orang
4. **Remove background** semua foto (save as PNG)
5. **Annotasi hati-hati** (bounding box shirt only)

---

## 🛠️ Tools yang Dibutuhkan

### Foto:
- Smartphone camera (rear camera, highest res)
- Atau webcam (min 720p)

### Background Removal:
```bash
pip install rembg
rembg i input.jpg output.png
```
Atau: remove.bg, Photoshop

### Annotation:
- LabelImg (https://github.com/heartexlabs/labelImg)
- CVAT (https://cvat.org)

### Validation:
```bash
python validate_dataset_quality.py
```

---

## 📈 Expected Improvement

### Dataset Lama (69 images):
- ❌ Accuracy: ~70%
- ❌ Fails pada: Pose baru, pencahayaan beda, orang baru

### Dataset Baru (300-400 images):
- ✅ Accuracy: ~85-90%
- ✅ Works pada: Berbagai pose, pencahayaan, orang baru
- ✅ Much more robust!

---

## ⏱️ Timeline

### Planning (1 jam)
- Recruit 5 volunteers
- Prepare 5 shirts (different colors)
- Setup space & camera

### Photoshoot (10 jam)
- 5 people × 2 hours = 10 hours
- (Bisa split ke 2 hari: 3 orang hari 1, 2 orang hari 2)

### Background Removal (5 jam)
- Batch process 400 images

### Annotation (6 hours)
- Annotate 400 images
- ~1 minute per image

### Validation & Cleanup (2 hours)
- Run validation script
- Fix issues

**Total: 24 jam = 3 hari kerja**

---

## 🎯 Minimum Viable Dataset (1 Hari)

Kalau **sangat terbatas waktu:**

### 4 Orang × 50 Foto = 200 Images

**Per orang (1 jam):**
1. Frontal poses (20 foto)
2. Rotations (15 foto)
3. Arms variations (10 foto)
4. Seated (5 foto)

**Hanya pakai 2-3 warna baju**

**Ini adalah MINIMUM ABSOLUT untuk improvement!**

---

## 📞 Quick Decision Tree

### Q: Berapa orang minimal?
**A:** 3-5 orang (lebih banyak = lebih baik)

### Q: Berapa foto per orang?
**A:** 50-80 foto (variasi pose)

### Q: Warna baju apa?
**A:** Putih (must!), Hitam, + 3 warna lain

### Q: Pose apa yang paling penting?
**A:** 
1. Frontal standing (40%)
2. Rotations ±30° (30%)
3. Arms raised/crossed (20%)
4. Seated (10%)

### Q: Berapa lama?
**A:** 
- Minimum: 1 hari (200 images)
- Recommended: 2-3 hari (400 images)

---

## ✅ Start Checklist

**Sebelum mulai, pastikan punya:**
- [ ] 5 volunteers confirmed
- [ ] 5 shirts ready (white, black, + 3 colors)
- [ ] Camera/phone ready
- [ ] Neutral background setup
- [ ] Good lighting
- [ ] This guide printed!
- [ ] Background removal tool installed
- [ ] Annotation tool installed

**Siap? Let's shoot! 📸**

---

*Untuk detail lengkap, baca: DATASET_COLLECTION_GUIDE.md*
*Untuk visual reference, baca: POSE_REFERENCE_SHEET.md*
