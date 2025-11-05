"""
Dataset Quality Validator
=========================

Script untuk validasi kualitas dataset sebelum training:
- Check jumlah images
- Check diversity (colors, poses, subjects)
- Check annotation quality
- Check image quality
- Generate statistics report
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from collections import Counter
import json


class DatasetValidator:
    def __init__(self, dataset_dir, annotations_csv):
        self.dataset_dir = Path(dataset_dir)
        self.annotations_path = Path(annotations_csv)
        self.report = {}
        
    def load_annotations(self):
        """Load annotations CSV"""
        try:
            self.df = pd.read_csv(self.annotations_path)
            print(f"✓ Loaded {len(self.df)} annotations")
            return True
        except Exception as e:
            print(f"✗ Failed to load annotations: {e}")
            return False
    
    def check_image_count(self):
        """Check total number of images"""
        total = len(self.df)
        
        status = "❌ TOO FEW"
        if total >= 500:
            status = "✅ EXCELLENT"
        elif total >= 300:
            status = "✅ GOOD"
        elif total >= 200:
            status = "⚠️  ACCEPTABLE"
        elif total >= 100:
            status = "⚠️  MINIMUM"
        
        self.report['total_images'] = {
            'count': total,
            'status': status,
            'recommendation': self._get_count_recommendation(total)
        }
        
        print(f"\n📊 Total Images: {total} {status}")
        print(f"   {self._get_count_recommendation(total)}")
        
    def _get_count_recommendation(self, count):
        if count >= 500:
            return "Dataset size is excellent!"
        elif count >= 300:
            return "Dataset size is good for training."
        elif count >= 200:
            return "Minimum viable, but more data recommended."
        elif count >= 100:
            return "CRITICAL: Need at least 200 images for decent accuracy."
        else:
            return "URGENT: Dataset too small. Collect more data ASAP!"
    
    def check_subject_diversity(self):
        """Check number of different subjects"""
        # Try to extract subject info from filenames
        subjects = set()
        for filename in self.df['filename']:
            # Assuming format: subjectXX_...
            parts = Path(filename).stem.split('_')
            if parts[0].startswith('subject') or parts[0].startswith('WIN'):
                subjects.add(parts[0])
        
        count = len(subjects) if subjects else 1  # At least 1
        
        status = "❌ POOR DIVERSITY"
        if count >= 5:
            status = "✅ GOOD"
        elif count >= 3:
            status = "⚠️  ACCEPTABLE"
        
        self.report['subject_diversity'] = {
            'count': count,
            'subjects': list(subjects)[:10],  # First 10
            'status': status
        }
        
        print(f"\n👥 Subject Diversity: {count} subjects {status}")
        if count < 3:
            print(f"   ⚠️  Recommend: Add 3-5 different people for better generalization")
    
    def check_color_diversity(self):
        """Estimate shirt color diversity from images"""
        print(f"\n🎨 Checking color diversity...")
        
        colors_detected = []
        sample_size = min(50, len(self.df))  # Sample up to 50 images
        
        for idx in np.random.choice(len(self.df), sample_size, replace=False):
            row = self.df.iloc[idx]
            img_path = self.dataset_dir / row['filename']
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Extract shirt region
                    x1, y1 = int(row['x_min']), int(row['y_min'])
                    x2, y2 = int(row['x_max']), int(row['y_max'])
                    
                    shirt_region = img[y1:y2, x1:x2]
                    
                    # Get dominant color
                    avg_color = np.mean(shirt_region, axis=(0, 1))
                    color_name = self._classify_color(avg_color)
                    colors_detected.append(color_name)
        
        color_counts = Counter(colors_detected)
        unique_colors = len(color_counts)
        
        status = "❌ POOR"
        if unique_colors >= 5:
            status = "✅ GOOD"
        elif unique_colors >= 3:
            status = "⚠️  ACCEPTABLE"
        
        self.report['color_diversity'] = {
            'unique_colors': unique_colors,
            'distribution': dict(color_counts),
            'status': status
        }
        
        print(f"   Colors detected: {unique_colors} unique colors {status}")
        print(f"   Distribution: {dict(color_counts)}")
        
        if unique_colors < 3:
            print(f"   ⚠️  Recommend: Use at least 5 different shirt colors")
    
    def _classify_color(self, bgr_color):
        """Simple color classification"""
        b, g, r = bgr_color
        
        # Check grayscale
        if max(r, g, b) - min(r, g, b) < 30:
            if np.mean([r, g, b]) < 50:
                return "black"
            elif np.mean([r, g, b]) > 200:
                return "white"
            else:
                return "gray"
        
        # Check colors
        if r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 180 and g > 180:
            return "yellow"
        else:
            return "other"
    
    def check_annotation_quality(self):
        """Check if annotations are reasonable"""
        print(f"\n📐 Checking annotation quality...")
        
        issues = []
        
        for idx, row in self.df.iterrows():
            x1, y1 = row['x_min'], row['y_min']
            x2, y2 = row['x_max'], row['y_max']
            
            width = x2 - x1
            height = y2 - y1
            
            # Check for invalid boxes
            if width <= 0 or height <= 0:
                issues.append(f"Row {idx}: Invalid box size (width={width}, height={height})")
            
            # Check for unreasonably small boxes
            if width < 50 or height < 50:
                issues.append(f"Row {idx}: Box too small (might not be shirt)")
            
            # Check for unreasonably large boxes
            if width > 1000 or height > 1000:
                issues.append(f"Row {idx}: Box suspiciously large")
            
            # Check aspect ratio (shirt should be roughly portrait oriented)
            aspect_ratio = height / width if width > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                issues.append(f"Row {idx}: Unusual aspect ratio ({aspect_ratio:.2f})")
        
        status = "✅ GOOD" if len(issues) < len(self.df) * 0.05 else "⚠️  HAS ISSUES"
        
        self.report['annotation_quality'] = {
            'total_checked': len(self.df),
            'issues_found': len(issues),
            'issues': issues[:10],  # First 10 issues
            'status': status
        }
        
        print(f"   Issues found: {len(issues)} / {len(self.df)} annotations {status}")
        
        if issues:
            print(f"   Sample issues:")
            for issue in issues[:5]:
                print(f"     - {issue}")
    
    def check_image_quality(self):
        """Check image resolution and quality"""
        print(f"\n🖼️  Checking image quality...")
        
        sample_size = min(20, len(self.df))
        resolutions = []
        missing = []
        
        for idx in np.random.choice(len(self.df), sample_size, replace=False):
            row = self.df.iloc[idx]
            img_path = self.dataset_dir / row['filename']
            
            if not img_path.exists():
                missing.append(str(img_path))
                continue
            
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                resolutions.append((w, h))
        
        if resolutions:
            avg_width = np.mean([r[0] for r in resolutions])
            avg_height = np.mean([r[1] for r in resolutions])
            
            status = "✅ GOOD"
            if avg_width < 640 or avg_height < 480:
                status = "⚠️  LOW RESOLUTION"
            
            self.report['image_quality'] = {
                'avg_resolution': f"{avg_width:.0f}x{avg_height:.0f}",
                'missing_files': len(missing),
                'status': status
            }
            
            print(f"   Average resolution: {avg_width:.0f}x{avg_height:.0f} {status}")
            print(f"   Missing files: {len(missing)}")
            
            if missing:
                print(f"   Sample missing files:")
                for f in missing[:3]:
                    print(f"     - {f}")
        else:
            print(f"   ⚠️  Could not load any images!")
    
    def check_pose_diversity(self):
        """Estimate pose diversity from bbox variations"""
        print(f"\n🤸 Checking pose diversity...")
        
        # Calculate bbox size variations
        widths = self.df['x_max'] - self.df['x_min']
        heights = self.df['y_max'] - self.df['y_min']
        
        width_std = np.std(widths)
        height_std = np.std(heights)
        
        # High variation = likely different poses/distances
        variation_score = (width_std + height_std) / 2
        
        status = "⚠️  LOW VARIETY"
        if variation_score > 100:
            status = "✅ GOOD VARIETY"
        elif variation_score > 50:
            status = "⚠️  SOME VARIETY"
        
        self.report['pose_diversity'] = {
            'bbox_variation': variation_score,
            'status': status
        }
        
        print(f"   Bounding box variation: {variation_score:.1f} {status}")
        print(f"   (Higher = more pose/distance diversity)")
        
        if variation_score < 50:
            print(f"   ⚠️  Recommend: Add more pose variations (arms raised, rotations, etc.)")
    
    def generate_summary_report(self):
        """Generate and print summary report"""
        print(f"\n" + "="*60)
        print("DATASET QUALITY SUMMARY")
        print("="*60)
        
        # Overall score
        scores = []
        for key, value in self.report.items():
            if 'status' in value:
                if '✅' in value['status']:
                    scores.append(100)
                elif '⚠️' in value['status']:
                    scores.append(50)
                else:
                    scores.append(0)
        
        overall_score = np.mean(scores) if scores else 0
        
        print(f"\n📊 Overall Quality Score: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            print("✅ EXCELLENT - Dataset is ready for training!")
        elif overall_score >= 60:
            print("✅ GOOD - Dataset is acceptable, but could be improved")
        elif overall_score >= 40:
            print("⚠️  FAIR - Dataset needs improvement for best results")
        else:
            print("❌ POOR - Dataset needs significant improvement")
        
        # Recommendations
        print(f"\n📝 Recommendations:")
        
        if self.report['total_images']['count'] < 300:
            print(f"   • Collect more images (target: 300-500)")
        
        if 'subject_diversity' in self.report and self.report['subject_diversity']['count'] < 3:
            print(f"   • Add more subjects (target: 5+)")
        
        if 'color_diversity' in self.report and self.report['color_diversity']['unique_colors'] < 5:
            print(f"   • Use more shirt colors (target: 5+)")
        
        if 'pose_diversity' in self.report and '⚠️' in self.report['pose_diversity']['status']:
            print(f"   • Add more pose variations")
        
        if 'annotation_quality' in self.report and self.report['annotation_quality']['issues_found'] > 0:
            print(f"   • Fix annotation issues ({self.report['annotation_quality']['issues_found']} found)")
        
        # Save report to JSON
        report_path = self.dataset_dir / "quality_report.json"
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types
            json_report = self._convert_to_json_serializable(self.report)
            json.dump(json_report, f, indent=2)
        
        print(f"\n✓ Full report saved to: {report_path}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def run_full_validation(self):
        """Run all validation checks"""
        print("="*60)
        print("DATASET QUALITY VALIDATION")
        print("="*60)
        
        if not self.load_annotations():
            return
        
        self.check_image_count()
        self.check_subject_diversity()
        self.check_color_diversity()
        self.check_annotation_quality()
        self.check_image_quality()
        self.check_pose_diversity()
        self.generate_summary_report()


def main():
    """Main entry point"""
    # Configuration
    DATASET_DIR = Path(__file__).parent / "IcanDataset_NOBG"
    ANNOTATIONS_CSV = Path(__file__).parent / "IcanDataset_annotations.csv"
    
    # Run validation
    validator = DatasetValidator(DATASET_DIR, ANNOTATIONS_CSV)
    validator.run_full_validation()


if __name__ == "__main__":
    main()
