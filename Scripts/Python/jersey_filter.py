import cv2
import numpy as np

class JerseyOverlay:
    def __init__(self):
        # Initialize a simple person detector (HOG + SVM) for torso estimation
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # Face detector for heuristic torso box (below face)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.face_cascade = None
        # Tracking & scheduling
        self.track_window = None  # (x, y, w, h)
        self.roi_hist = None      # HSV histogram for CamShift backprojection
        self.frame_idx = 0
        self.redetect_interval = 20   # frames between full re-detections
        self.refine_interval = 6      # frames between expensive GrabCut refinements
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        # Jersey system
        self.current_jersey = "none"
        self.jerseys = self.create_jersey_templates()
    
    def create_jersey_templates(self):
        """Create jersey templates for various clubs"""
        jerseys = {}
        
        def create_jersey_template(base_color, name, pattern_type="solid"):
            # Create a 200x300 jersey template
            jersey = np.zeros((300, 200, 3), dtype=np.uint8)
            jersey[:] = base_color
            
            if name == "brighton":
                # Brighton & Hove Albion - Blue and white stripes
                stripe_width = 25
                for i in range(0, 200, stripe_width * 2):
                    # White stripes
                    cv2.rectangle(jersey, (i, 0), (i + stripe_width, 300), (255, 255, 255), -1)
                # Add simple collar
                cv2.rectangle(jersey, (10, 10), (190, 40), (0, 100, 200), -1)
                # Add sponsor text area (subtle)
                cv2.rectangle(jersey, (50, 120), (150, 150), (200, 200, 200), 1)
                
            elif name == "brazil":
                # Yellow with green trim
                cv2.rectangle(jersey, (10, 10), (190, 50), (0, 180, 0), -1)  # Green collar
                cv2.putText(jersey, "10", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 100, 255), 5)
                
            elif name == "argentina":
                # Light blue and white stripes
                stripe_height = 30
                for i in range(0, 300, stripe_height * 2):
                    cv2.rectangle(jersey, (0, i), (200, i + stripe_height), (255, 255, 255), -1)
                    
            elif name == "germany":
                # White with black details
                cv2.rectangle(jersey, (10, 10), (190, 50), (0, 0, 0), -1)  # Black collar
                cv2.putText(jersey, "11", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
            
            return jersey
        
        # Create different jersey templates
        jerseys["brighton"] = create_jersey_template((0, 100, 200), "brighton")  # Blue base
        jerseys["brazil"] = create_jersey_template((0, 255, 255), "brazil")      # Yellow
        jerseys["argentina"] = create_jersey_template((255, 200, 150), "argentina") # Light blue
        jerseys["germany"] = create_jersey_template((255, 255, 255), "germany")     # White
        jerseys["none"] = None
        
        return jerseys
    
    def set_jersey(self, jersey_name):
        """Change the current jersey"""
        if jersey_name in self.jerseys:
            self.current_jersey = jersey_name
            return True
        return False
    
    def get_available_jerseys(self):
        """Get list of available jersey names"""
        return list(self.jerseys.keys())
    
    def detect_torso_area(self, frame):
        """Detect torso region. Uses HOG person detector first, falls back to HSV heuristic.
        Returns: (torso_mask, (x1, y1, x2, y2))
        """
        height, width = frame.shape[:2]

        # 1) Try HOG person detection for a robust bounding box
        bbox = None
        try:
            # Work on a smaller copy for speed
            scale = 400.0 / max(1.0, max(height, width))
            if scale < 1.0:
                small = cv2.resize(frame, (int(width * scale), int(height * scale)))
            else:
                small = frame
                scale = 1.0

            rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
            # Choose the strongest/ largest detection
            best_idx = -1
            best_score = -1.0
            for i, (x, y, w, h) in enumerate(rects):
                score = float(weights[i]) if i < len(weights) else 0.0
                area = w * h
                # Prefer larger and higher score
                score = score + 0.000001 * area
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx >= 0:
                x, y, w, h = rects[best_idx]
                # Map back to original scale
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
                # Define torso region roughly between shoulders and hips
                x1 = max(0, x + int(0.15 * w))
                x2 = min(width, x + int(0.85 * w))
                y1 = max(0, y + int(0.20 * h))
                y2 = min(height, y + int(0.75 * h))
                if x2 > x1 and y2 > y1:
                    bbox = (x1, y1, x2, y2)
        except Exception:
            # If HOG fails for any reason, we'll fall back below
            bbox = None

        if bbox is None and self.face_cascade is not None:
            # 2) Fallback: detect face, then infer torso box under the face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                # Choose the largest face (closest person)
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                # Heuristic torso box: centered at face center, below the face
                cx = fx + fw // 2
                # Width: widen beyond face width; Height: extend to mid-torso
                t_w = int(fw * 2.0)
                t_h = int(fh * 2.2)
                x1 = max(0, cx - t_w // 2)
                y1 = max(0, fy + int(1.0 * fh))  # start just under the chin
                x2 = min(width, x1 + t_w)
                y2 = min(height, y1 + t_h)
                if x2 > x1 and y2 > y1:
                    bbox = (x1, y1, x2, y2)

        if bbox is None:
            # 3) Final fallback: central rectangle likely covering upper body
            start_y = int(height * 0.25)
            end_y = int(height * 0.80)
            start_x = int(width * 0.25)
            end_x = int(width * 0.75)
            bbox = (start_x, start_y, end_x, end_y)

        # If we have a bbox from HOG, create a simple rectangular mask
        x1, y1, x2, y2 = bbox
        torso_mask = np.zeros((height, width), dtype=np.uint8)
        torso_mask[y1:y2, x1:x2] = 255
        return torso_mask, bbox

    def visualize_torso(self, frame, torso_mask, bbox, color=(255, 0, 0), alpha=0.5):
        """Overlay the detected torso area for debugging/preview."""
        x1, y1, x2, y2 = bbox
        overlay = frame.copy()
        # Color overlay only within mask
        colored = np.zeros_like(frame)
        colored[:] = color
        mask3 = cv2.merge([torso_mask, torso_mask, torso_mask])
        overlay = np.where(mask3 > 0, (overlay * (1 - alpha) + colored * alpha).astype(np.uint8), overlay)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return overlay

    def _init_tracker(self, frame, bbox):
        """Initialize CamShift tracker with a histogram from the chest area inside bbox."""
        x1, y1, x2, y2 = bbox
        x = max(0, x1)
        y = max(0, y1)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        self.track_window = (x, y, w, h)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Chest subregion
        cx1 = x + int(0.2 * w)
        cx2 = x + int(0.8 * w)
        cy1 = y + int(0.15 * h)
        cy2 = y + int(0.55 * h)
        cx1 = max(0, min(frame.shape[1] - 1, cx1))
        cx2 = max(0, min(frame.shape[1], cx2))
        cy1 = max(0, min(frame.shape[0] - 1, cy1))
        cy2 = max(0, min(frame.shape[0], cy2))
        roi = hsv[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            self.roi_hist = None
            return
        # Mask out likely skin in ROI when building hist
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin = cv2.inRange(roi, lower_skin, upper_skin)
        mask = cv2.bitwise_not(skin)
        # Hue-Sat histogram (ignore value)
        self.roi_hist = cv2.calcHist([roi], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

    def _update_tracker(self, frame):
        """Update CamShift tracker; returns updated bbox or None if not available."""
        if self.roi_hist is None or self.track_window is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)
        # Limit search to a belt around the current window to reduce drift
        x, y, w, h = self.track_window
        pad = int(max(10, 0.25 * max(w, h)))
        sx = max(0, x - pad)
        sy = max(0, y - pad)
        ex = min(frame.shape[1], x + w + pad)
        ey = min(frame.shape[0], y + h + pad)
        sub_bp = back_proj[sy:ey, sx:ex]
        local_window = (x - sx, y - sy, w, h)
        try:
            ret, local_window = cv2.CamShift(sub_bp, local_window, self.term_crit)
            # Map back to full image coordinates
            lx, ly, lw, lh = local_window
            self.track_window = (sx + lx, sy + ly, lw, lh)
        except Exception:
            return None
        x, y, w, h = self.track_window
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _refine_torso_mask_with_grabcut(self, frame, bbox):
        """Refine torso mask using GrabCut, seeded by neck/chest area and excluding face/skin.
        Returns binary mask (uint8 0/255)."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        # Estimate neck line using face if available
        neck_y = y1 + int(0.10 * (y2 - y1))
        try:
            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                    neck_y = min(height - 1, fy + int(1.0 * fh))
        except Exception:
            pass

        roi_y1 = max(0, neck_y - 5)
        roi_y2 = min(height, y2 + int(0.15 * (y2 - y1)))
        roi_x1 = 0
        roi_x2 = width
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_h, roi_w = roi.shape[:2]
        if roi_h <= 0 or roi_w <= 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            return mask

        # Initialize GrabCut mask
        GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = 0, 1, 2, 3
        gc_mask = np.full((roi_h, roi_w), GC_PR_BGD, dtype=np.uint8)

        # Seed background: skin (face/neck/hands) using HSV skin range
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin = cv2.inRange(hsv_roi, lower_skin, upper_skin)
        skin = cv2.medianBlur(skin, 5)
        gc_mask[skin > 0] = GC_BGD

        # Hard background above neck line strip
        strip_h = max(5, roi_h // 20)
        gc_mask[:strip_h, :] = GC_BGD

        # Seed foreground: central chest region
        cx1 = int(roi_w * 0.35)
        cx2 = int(roi_w * 0.65)
        cy1 = int(roi_h * 0.15)
        cy2 = int(roi_h * 0.55)
        gc_mask[cy1:cy2, cx1:cx2] = GC_FGD

        # Expand probable foreground by color similarity around chest seed
        chest = roi[cy1:cy2, cx1:cx2]
        if chest.size > 0:
            mean_bgr = chest.reshape(-1, 3).mean(axis=0)
            diff = np.linalg.norm(roi.astype(np.float32) - mean_bgr.astype(np.float32), axis=2)
            thresh = max(20.0, float(diff.mean() * 0.6))
            pr_fgd = (diff < thresh).astype(np.uint8)
            gc_mask[pr_fgd > 0] = np.where(gc_mask[pr_fgd > 0] == GC_BGD, GC_BGD, GC_PR_FGD)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(roi, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
        except Exception:
            # On failure, fallback to rectangle mask
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            return mask

        # Binary mask from GrabCut result
        mask_roi = np.where((gc_mask == GC_FGD) | (gc_mask == GC_PR_FGD), 255, 0).astype(np.uint8)

        # Post-process: remove small areas, close holes, keep biggest component touching chest seed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)

        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_roi, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_roi = np.where(labels == largest, 255, 0).astype(np.uint8)

        # Paste back into full-size mask
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = mask_roi
        return full_mask
    
    def apply_jersey_overlay(self, frame):
        """Detect and visualize torso only (optimized for FPS)."""
        self.frame_idx += 1

        # Decide whether to re-detect from scratch
        do_redetect = (self.track_window is None) or (self.frame_idx % self.redetect_interval == 1)
        if do_redetect:
            torso_mask_rect, (start_x, start_y, end_x, end_y) = self.detect_torso_area(frame)
            bbox = (start_x, start_y, end_x, end_y)
            self._init_tracker(frame, bbox)
        else:
            bbox = self._update_tracker(frame)
            if bbox is None:
                torso_mask_rect, (start_x, start_y, end_x, end_y) = self.detect_torso_area(frame)
                bbox = (start_x, start_y, end_x, end_y)
                self._init_tracker(frame, bbox)
            # Build a quick rectangular mask from tracker bbox
            height, width = frame.shape[:2]
            torso_mask_rect = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            torso_mask_rect[y1:y2, x1:x2] = 255

        # Determine whether to run expensive GrabCut refinement this frame
        run_refine = (self.frame_idx % self.refine_interval == 0)
        if run_refine:
            refined_mask = self._refine_torso_mask_with_grabcut(frame, bbox)
            if refined_mask.sum() < 1000:
                refined_mask = torso_mask_rect
        else:
            # Fast mask using backprojection threshold inside bbox
            if self.roi_hist is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                back_proj = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)
                x1, y1, x2, y2 = bbox
                bp = back_proj[y1:y2, x1:x2]
                _, bp_bin = cv2.threshold(bp, 40, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                bp_bin = cv2.morphologyEx(bp_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
                bp_bin = cv2.morphologyEx(bp_bin, cv2.MORPH_OPEN, kernel, iterations=1)
                refined_mask = np.zeros_like(torso_mask_rect)
                refined_mask[y1:y2, x1:x2] = bp_bin
                if refined_mask.sum() < 1000:
                    refined_mask = torso_mask_rect
            else:
                refined_mask = torso_mask_rect

        # Apply jersey overlay if selected
        if self.current_jersey != "none" and self.current_jersey in self.jerseys:
            jersey_template = self.jerseys[self.current_jersey]
            if jersey_template is not None:
                frame = self._blend_jersey(frame, refined_mask, bbox, jersey_template)
        else:
            # Just show torso visualization (debug)
            frame = self.visualize_torso(frame, refined_mask, bbox)
        
        return frame
    
    def _blend_jersey(self, frame, torso_mask, bbox, jersey_template):
        """Blend jersey template onto the torso area."""
        x1, y1, x2, y2 = bbox
        torso_width = x2 - x1
        torso_height = y2 - y1
        
        if torso_width <= 0 or torso_height <= 0:
            return frame
        
        # Resize jersey template to fit torso region
        jersey_resized = cv2.resize(jersey_template, (torso_width, torso_height))
        
        # Blend with transparency
        alpha = 0.7  # Jersey opacity
        
        overlay_region = frame[y1:y2, x1:x2].copy()
        torso_mask_region = torso_mask[y1:y2, x1:x2]
        
        # Apply jersey where mask is active
        for c in range(3):  # BGR channels
            overlay_region[:, :, c] = np.where(
                torso_mask_region > 0,
                overlay_region[:, :, c] * (1 - alpha) + jersey_resized[:, :, c] * alpha,
                overlay_region[:, :, c]
            )
        
        frame[y1:y2, x1:x2] = overlay_region
        
        # Draw bbox for reference
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame