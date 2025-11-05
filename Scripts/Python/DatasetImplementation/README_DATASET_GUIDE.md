# Dataset Implementation Guide

## 📚 Complete Documentation Suite

This folder contains all documentation and tools for collecting and improving the jersey detection dataset.

---

## 📖 Documentation Files

### 1. **DATASET_SUMMARY.md** ⭐ START HERE!
**Quick reference guide - read this first!**
- TL;DR of what poses to collect
- Minimum viable dataset (200 images)
- Recommended dataset (400 images)
- Quick checklists

**Read this if:** You want quick answers and don't have time for details.

---

### 2. **DATASET_COLLECTION_GUIDE.md** 📘 COMPREHENSIVE
**Complete, detailed guide for dataset collection**
- Detailed pose categories with priority ratings
- Subject diversity requirements
- Lighting and camera setup guidelines
- Step-by-step collection process
- Tools and resources
- Quality checklist

**Read this if:** You want to understand everything in depth before starting.

---

### 3. **POSE_REFERENCE_SHEET.md** 📸 PRACTICAL
**Print-friendly pose reference for photoshoot**
- Visual pose diagrams (ASCII art)
- Priority tiers (Tier 1, 2, 3)
- Photo count per pose
- Time estimates
- Checklist for each session
- Photography tips

**Use this during:** Actual photoshoot - print and bring it!

---

### 4. **VISUAL_POSE_GUIDE.txt** 🎨 VISUAL
**Visual diagrams and matrices**
- Pose priority matrix
- Dataset composition charts
- Timeline visualization
- Subject diversity requirements
- Annotation guide (what to include/exclude)
- Quality scorecard

**Use this for:** Visual understanding and planning.

---

## 🛠️ Tools & Scripts

### 5. **validate_dataset_quality.py** ✅ VALIDATOR
**Automated dataset quality checker**

```bash
python validate_dataset_quality.py
```

**Features:**
- Check total image count
- Analyze subject diversity
- Estimate color diversity
- Validate annotation quality
- Check image resolution
- Estimate pose diversity
- Generate quality report (JSON)

**Run this:** Before training to ensure dataset quality!

---

## 🚀 Quick Start Guide

### Current Situation:
- **Your dataset:** ~69 images, 1-2 subjects, limited poses
- **Problem:** Model accuracy ~70%, not robust to new people/poses
- **Solution:** Collect 300-400 images with diversity!

---

## 📝 3-Step Action Plan

### Step 1: READ (30 minutes)
1. **Start here:** Read **DATASET_SUMMARY.md**
2. **For details:** Skim **DATASET_COLLECTION_GUIDE.md**
3. **Print this:** **POSE_REFERENCE_SHEET.md**

### Step 2: COLLECT (2-3 days)
1. **Recruit:** 5 volunteers (different skin tones, body types)
2. **Prepare:** 5 shirts per person (white, black, + 3 colors)
3. **Shoot:** 80 photos per person = 400 photos total
   - Use POSE_REFERENCE_SHEET as checklist
   - Follow priority order (Tier 1 → Tier 2 → Tier 3)

### Step 3: PROCESS & VALIDATE (1 day)
1. **Remove backgrounds:** Use rembg or remove.bg
2. **Annotate:** Use LabelImg to mark shirt regions
3. **Validate:** Run `python validate_dataset_quality.py`
4. **Retrain:** Run `python shirt_detector_full.py`

---

## 🎯 Pose Priorities (Quick Reference)

### TIER 1 - CRITICAL (Must have! 60% of dataset):

1. **Frontal standing** (hands sides, on hips, crossed) - 40 photos
2. **Quarter rotations** (±30° left/right) - 20 photos

### TIER 2 - IMPORTANT (30% of dataset):

3. **Arms raised** (one arm, both arms, gestures) - 15 photos
4. **Seated poses** - 10 photos

### TIER 3 - NICE TO HAVE (10% of dataset):

5. **Side profiles, leaning, edge cases** - 10 photos

**Total: ~80 photos per person**

---

## 👥 Diversity Requirements

### Must Have:
- ✅ **5+ people** (not just 1-2!)
- ✅ **3+ skin tones** (light, medium, dark)
- ✅ **5+ shirt colors** (white, black, red, blue, green)
- ✅ **3 lighting conditions** (bright, normal, dim)

### Why This Matters:
- **More people:** Model learns general "shirt" concept, not just one person's shirt
- **More colors:** Model doesn't overfit to one color
- **More poses:** Model works in real-world scenarios
- **More lighting:** Model robust to different conditions

---

## 📊 Expected Results

### Before (69 images):
- ❌ Accuracy: ~70%
- ❌ Fails on: New people, unusual poses, different lighting
- ❌ Robustness: Low

### After (400 images with diversity):
- ✅ Accuracy: ~85-90%
- ✅ Works on: Different people, various poses, multiple conditions
- ✅ Robustness: High

**Improvement: ~15-20% better accuracy + much more reliable!**

---

## ⏱️ Time Investment

### Minimum Viable (1 day):
- **Target:** 200 images (4 people × 50 photos)
- **Time:** 10 hours total
- **Result:** Noticeable improvement

### Recommended (2-3 days):
- **Target:** 400 images (5 people × 80 photos)
- **Time:** 21 hours total
- **Result:** Significant improvement

**Worth it?** YES! 2-3 days of work = much better model for entire project!

---

## 🛠️ Tools You Need

### For Photoshoot:
- Smartphone camera (rear camera, high resolution)
- Neutral background (white wall or bedsheet)
- Good lighting (near window or room lights)

### For Processing:
```bash
# Background removal
pip install rembg
rembg i input.jpg output.png

# Or use: remove.bg website
```

### For Annotation:
- **LabelImg:** https://github.com/heartexlabs/labelImg
- Free, easy to use, saves to CSV format

---

## ✅ Quality Checklist

Before calling dataset "done":

- [ ] Total images: 300+ ✓
- [ ] Different subjects: 5+ ✓
- [ ] Shirt colors per subject: 5+ ✓
- [ ] Pose variations: 15+ ✓
- [ ] Lighting conditions: 3+ ✓
- [ ] Backgrounds removed: 100% ✓
- [ ] Annotations complete: 100% ✓
- [ ] Validation score: >80% ✓

---

## 🚨 Common Mistakes to Avoid

### ❌ DON'T:
1. **Only collect from 1-2 people** → Model won't work on others
2. **Only use 1-2 shirt colors** → Model overfits to color
3. **Only frontal poses** → Model fails on rotations
4. **Skip background removal** → Training quality suffers
5. **Rush annotations** → Bad labels = bad training

### ✅ DO:
1. **5+ diverse subjects** → Model generalizes
2. **5+ colors per subject** → Color-independent detection
3. **15+ pose variations** → Works in real scenarios
4. **Clean preprocessing** → Better training data
5. **Careful annotation** → Accurate labels

---

## 📁 File Organization

Recommended structure for new dataset:

```
NewDataset/
├── subject01/
│   ├── white/
│   │   ├── frontal_001.png
│   │   ├── frontal_002.png
│   │   └── ...
│   ├── black/
│   └── red/
├── subject02/
├── subject03/
├── subject04/
├── subject05/
└── annotations.csv
```

---

## 💡 Pro Tips

### During Photoshoot:
1. **Take 2-3 photos per pose** (pick best later)
2. **Use timer/self-timer** (avoid camera shake)
3. **Check focus** after each session
4. **Take breaks** (avoid fatigue)
5. **Keep lighting consistent** within each session

### During Processing:
1. **Batch process** background removal (saves time)
2. **Standardize filenames** (easier to manage)
3. **Double-check annotations** (prevent training issues)
4. **Run validation** before training (catch issues early)

---

## 🎓 Learning Points

### Why is current dataset insufficient?

**Current (69 images, 1-2 people):**
```
Model learns: "This specific person's shirt looks like THIS"
Problem: Doesn't generalize to other people
```

**Improved (400 images, 5+ people):**
```
Model learns: "Shirts in general look like THIS across different people"
Result: Works on anyone!
```

### Why diversity matters?

**Without diversity:**
- Model: "Shirt = blue colored object on THIS person"
- Fails when: Different color, different person, different pose

**With diversity:**
- Model: "Shirt = torso-worn fabric, various colors, on various people"
- Works when: Any color, any person, many poses!

---

## 📞 Quick FAQ

**Q: Berapa foto minimum yang butuh?**
**A:** Minimum 200, recommended 400.

**Q: Berapa orang minimal?**
**A:** Minimum 3, recommended 5+.

**Q: Warna baju apa yang harus dipakai?**
**A:** Putih (wajib!), hitam, + 3 warna lain (merah/biru/hijau).

**Q: Berapa lama prosesnya?**
**A:** 1-3 hari tergantung target (200 vs 400 images).

**Q: Apakah akan signifikan peningkatannya?**
**A:** YA! Dari ~70% accuracy ke ~85-90%. Plus jauh lebih robust.

**Q: Bisakah pakai 1 orang dengan banyak baju?**
**A:** Bisa, tapi lebih baik 5 orang dengan variety. Diversity > quantity dari 1 orang.

---

## 🚀 Next Steps - Action Items

### This Week:
1. [ ] Read DATASET_SUMMARY.md (15 min)
2. [ ] Recruit 5 volunteers
3. [ ] Gather 5+ shirts (different colors)
4. [ ] Print POSE_REFERENCE_SHEET.md

### Next Week:
5. [ ] Conduct photoshoots (2-3 days)
6. [ ] Process images (background removal)
7. [ ] Annotate images (LabelImg)
8. [ ] Validate dataset quality

### Following Week:
9. [ ] Retrain model with new data
10. [ ] Test and compare results
11. [ ] Document improvements

---

## 📚 Documentation Index

| File | Purpose | When to Use |
|------|---------|-------------|
| **DATASET_SUMMARY.md** | Quick reference | First read, quick lookup |
| **DATASET_COLLECTION_GUIDE.md** | Comprehensive guide | Planning phase |
| **POSE_REFERENCE_SHEET.md** | Photoshoot checklist | During photoshoot |
| **VISUAL_POSE_GUIDE.txt** | Visual diagrams | Understanding structure |
| **validate_dataset_quality.py** | Quality checker | After collection |

---

## 🎉 Success Criteria

Your dataset is ready when:

1. ✅ Validation score >80%
2. ✅ Model accuracy improves by 10-15%
3. ✅ Works on people not in training set
4. ✅ Works on various poses
5. ✅ Works in different lighting

---

**Ready to start? Begin with DATASET_SUMMARY.md! 🚀**

*Good luck with your dataset collection!*  
*Last updated: November 2025*
