from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..config.settings import Settings
import io

# Optional YOLO import (lazy)
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

# ------------------------------------------------------------------
# Ensure PyTorch safe serializer recognises Ultralytics classes *before*
# any model weights are loaded (multiprocess workers, no reload).
# ------------------------------------------------------------------
try:
    import torch.serialization as _ts  # type: ignore
    from ultralytics.nn.tasks import DetectionModel  # type: ignore
    from torch.nn.modules.container import Sequential  # type: ignore
    from ultralytics.nn.modules import Conv  # type: ignore
    if hasattr(_ts, "add_safe_globals"):
        # torch.serialization.add_safe_globals expects a *dict* that maps the fully
        # qualified name to the object. Passing a list (what we did before) is a
        # no-op, so YOLO kept failing on every new class that appeared in the
        # checkpoint. Register the needed classes properly so the weight file
        # loads once and stays in memory.
        _safe_dict = {
            f"{cls.__module__}.{cls.__qualname__}": cls for cls in (DetectionModel, Sequential, Conv)
        }
        _ts.add_safe_globals(_safe_dict)
except Exception:
    # If torch is missing or API changed we silently continue – loader will fallback.
    pass

# Optional Mask R-CNN import (torchvision)
try:
    import torch  # type: ignore
    from torchvision import models as tv_models, transforms as tv_transforms  # type: ignore
except Exception:
    torch = None
    tv_models = None

# Optional ONNX Runtime (for accelerated inference)
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

# Optional MiDaS depth-estimation via torch.hub
# We will lazy-load to avoid long startup times.
try:
    import torch  # type: ignore  # re-use if already imported
except Exception:
    torch = None

class ImageProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings

        # ------------------------------------------------------------
        # YOLO initialisation (PyTorch or ONNX depending on settings)
        # ------------------------------------------------------------
        self._yolo_model = None
        if self.settings.USE_YOLO_DETECTION and YOLO is not None:
            try:
                # Decide which model path to load
                model_path = (
                    self.settings.YOLO_ONNX_MODEL
                    if self.settings.USE_ONNX_ACCELERATION
                    else self.settings.YOLO_MODEL
                )

                if not self.settings.USE_ONNX_ACCELERATION:
                    # Allow ultralytics DetectionModel class in torch safe loader (PyTorch>=2.6)
                    try:
                        import torch.serialization as _ts  # type: ignore
                        from ultralytics.nn.tasks import DetectionModel  # type: ignore
                        from torch.nn.modules.container import Sequential  # type: ignore
                        from ultralytics.nn.modules import Conv  # type: ignore
                        if hasattr(_ts, "add_safe_globals"):
                            _ts.add_safe_globals({
                                "ultralytics.nn.tasks.DetectionModel": DetectionModel,
                                "torch.nn.modules.container.Sequential": Sequential,
                                "ultralytics.nn.modules.Conv": Conv,
                            })  # type: ignore[arg-type]
                    except Exception:
                        pass

                # Loading via ultralytics automatically chooses backend based on extension
                self._yolo_model = YOLO(model_path)
            except Exception as e:
                # Disable YOLO on failure but keep server running
                self.settings.USE_YOLO_DETECTION = False
                print(f"[WARN] Failed to load YOLO model: {e}. Falling back to OpenCV pipeline.")

        # ------------------------------------------------------------
        # Mask R-CNN initialisation (Torch or ONNX)
        # ------------------------------------------------------------
        self._mask_model = None  # torch version
        self._mask_session = None  # onnxruntime session

        if self.settings.USE_MASK_RCNN:
            # First preference: ONNX acceleration if requested and runtime present
            if self.settings.USE_ONNX_ACCELERATION and ort is not None:
                try:
                    providers = ort.get_available_providers()
                    self._mask_session = ort.InferenceSession(
                        self.settings.MASK_RCNN_ONNX_MODEL, providers=providers
                    )
                except Exception as e:
                    print(f"[WARN] Failed to load ONNX Mask R-CNN: {e}. Falling back to Torch model.")

            # If ONNX not used / failed, fall back to TorchVision model
            if self._mask_session is None and tv_models is not None:
                try:
                    self._mask_model = tv_models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
                    self._mask_model.eval()
                    if torch and torch.cuda.is_available():
                        self._mask_model.to("cuda")
                    self._mask_tf = tv_transforms.Compose([tv_transforms.ToTensor()])
                except Exception as e:
                    self.settings.USE_MASK_RCNN = False
                    print(f"[WARN] Failed to load Torch Mask R-CNN: {e}.")

        self.reference_sizes = {
            "Credit Card": {"width": 85.60, "height": 53.98},
            "Quarter (US)": {"diameter": 24.26},
            "Standard Plate": {"diameter": 250},
            "iPhone": {"width": 71.5, "height": 146.7}
        }

        # ------------------------------------------------------------
        # MiDaS depth-estimation initialisation (lazy)
        # ------------------------------------------------------------
        self._depth_model = None
        self._depth_tf = None
        if self.settings.USE_MIDAS_DEPTH and torch is not None:
            try:
                # Download/load model using torch.hub
                self._depth_model = torch.hub.load("intel-isl/MiDaS", self.settings.MIDAS_MODEL_NAME)
                self._depth_model.eval()

                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                if self.settings.MIDAS_MODEL_NAME.lower() == "midas_small":
                    self._depth_tf = midas_transforms.small_transform
                elif "dpt" in self.settings.MIDAS_MODEL_NAME.lower():
                    self._depth_tf = midas_transforms.dpt_transform
                else:
                    self._depth_tf = midas_transforms.default_transform

                if torch.cuda.is_available():
                    self._depth_model.to("cuda")
            except Exception as e:
                self.settings.USE_MIDAS_DEPTH = False
                print(f"[WARN] Failed to load MiDaS model: {e}. Depth estimation disabled.")

    def preprocess_image(self, image: Image.Image) -> Dict:
        """Enhanced image preprocessing with configurable quality settings"""
        try:
            # Convert PIL Image to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Automatic image compression
            img_cv = self._compress_image(img_cv)
            
            # Basic preprocessing
            if self.settings.PROCESSING_QUALITY == "fast":
                # Fast processing: just convert to grayscale
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
            elif self.settings.PROCESSING_QUALITY == "balanced":
                # Balanced processing: moderate enhancement
                lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_l = clahe.apply(l)
                enhanced_lab = cv2.merge([enhanced_l, a, b])
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
            else:  # quality mode
                # Full quality processing
                lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced_l = clahe.apply(l)
                enhanced_lab = cv2.merge([enhanced_l, a, b])
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                sigma = 0.33
                median = np.median(gray)
                lower = int(max(0, (1.0 - sigma) * median))
                upper = int(min(255, (1.0 + sigma) * median))
                edges = cv2.Canny(gray, lower, upper)
            
            # Clean up edges
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            cleaned_edges = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            
            return {
                'original': img_cv,
                'enhanced': enhanced if self.settings.PROCESSING_QUALITY != "fast" else img_cv,
                'edges': cleaned_edges,
                'dilated': dilated,
                'gray': gray
            }
        except Exception as e:
            raise ValueError(f"Error in image preprocessing: {str(e)}")

    def _compress_image(self, img: np.ndarray) -> np.ndarray:
        """Automatically compress image while maintaining quality"""
        # Get current image dimensions
        height, width = img.shape[:2]
        
        # Calculate target size while maintaining aspect ratio
        max_dimension = self.settings.MAX_IMAGE_SIZE
        scale = min(max_dimension / width, max_dimension / height)
        
        if scale < 1:  # Only resize if image is larger than max dimension
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Calculate compression quality based on image size
            if self.settings.PROCESSING_QUALITY == "fast":
                quality = 85
            elif self.settings.PROCESSING_QUALITY == "balanced":
                quality = 90
            else:  # quality mode
                quality = 95
                
            # Convert to PIL Image for compression
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Create a BytesIO object to store the compressed image
            buffer = io.BytesIO()
            
            # Save with compression
            pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
            
            # Read back the compressed image
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            
            # Convert back to OpenCV format
            img = cv2.cvtColor(np.array(compressed_img), cv2.COLOR_RGB2BGR)
        
        return img

    def detect_food_items(self, image: Image.Image, reference_object: Optional[str] = None) -> Dict:
        """Detect and analyze food items in the image"""

        # If YOLO enabled and model loaded, run it first
        if self.settings.USE_YOLO_DETECTION and self._yolo_model is not None:
            try:
                results = self._yolo_model.predict(np.array(image))
                food_instances = self._yolo_results_to_instances(results)
                return {
                    'food_instances': food_instances,
                    'visualization': image,
                    'calibration': None,
                    'preprocessed': None
                }
            except Exception as e:
                print(f"[WARN] YOLO detection failed: {e}. Falling back to OpenCV pipeline.")

        # If Mask R-CNN enabled (without YOLO) run segmentation-only pipeline
        if self.settings.USE_MASK_RCNN and (
            self._mask_session is not None or self._mask_model is not None
        ):
            # Branch 1: ONNX Runtime inference
            if self._mask_session is not None:
                try:
                    # Pre-process image to tensor shape (1,3,H,W) with float32 pixel values 0-1
                    img_np = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
                    img_np = np.transpose(img_np, (2, 0, 1))  # CHW
                    img_np = np.expand_dims(img_np, 0)

                    ort_inputs = {self._mask_session.get_inputs()[0].name: img_np}
                    ort_outs = self._mask_session.run(None, ort_inputs)

                    # Simple parsing – assume torchvision style outputs order
                    # [boxes, labels, scores, masks]
                    if len(ort_outs) >= 4:
                        outputs = {
                            "boxes": ort_outs[0],
                            "labels": ort_outs[1],
                            "scores": ort_outs[2],
                            "masks": ort_outs[3],
                        }
                        instances = self._mask_results_to_instances(outputs)
                        return {
                            'food_instances': instances,
                            'visualization': image,
                            'calibration': None,
                            'preprocessed': None
                        }
                except Exception as e:
                    print(f"[WARN] ONNX Mask R-CNN failed: {e}. Falling back to Torch/CPU pipeline.")

            # Branch 2: TorchVision Mask R-CNN
            if self._mask_model is not None:
                try:
                    with torch.no_grad():  # type: ignore[attr-defined]
                        img_tensor = self._mask_tf(image)
                        if torch and torch.cuda.is_available():
                            img_tensor = img_tensor.to("cuda")
                        outputs = self._mask_model([img_tensor])[0]
                    instances = self._mask_results_to_instances(outputs)
                    return {
                        'food_instances': instances,
                        'visualization': image,
                        'calibration': None,
                        'preprocessed': None
                    }
                except Exception as e:
                    print(f"[WARN] Mask R-CNN failed: {e}. Falling back to OpenCV pipeline.")

        # OpenCV fallback pipeline
        preprocessed = self.preprocess_image(image)
        if not preprocessed:
            return None
            
        # Perform watershed segmentation
        markers, segmented = self._watershed_segmentation(preprocessed)
        
        # Calibrate size if reference object is present
        calibration = None
        if reference_object:
            calibration = self._calibrate_size(image, reference_object, preprocessed)
        
        # Detect and analyze food items
        food_instances = self._analyze_food_instances(preprocessed, markers, calibration)
        
        # Create visualization
        result_img = self._create_visualization(preprocessed['enhanced'], food_instances, calibration)
        
        return {
            'food_instances': food_instances,
            'visualization': Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)),
            'calibration': calibration,
            'preprocessed': preprocessed
        }

    def _watershed_segmentation(self, preprocessed: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Perform watershed segmentation with quality-based optimization"""
        edges = preprocessed['edges']
        img = preprocessed['enhanced']
        
        if self.settings.PROCESSING_QUALITY == "fast":
            # Simplified segmentation for fast processing
            _, edges_binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3,3), np.uint8)
            sure_bg = cv2.dilate(edges_binary, kernel, iterations=2)
            markers = cv2.connectedComponents(sure_bg)[1]
            markers = markers + 1
            markers[edges_binary == 255] = 0
        else:
            # Full watershed segmentation
            _, edges_binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3,3), np.uint8)
            sure_bg = cv2.dilate(edges_binary, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(edges_binary, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(img, markers)
        
        segmented = img.copy()
        segmented[markers == -1] = [0, 0, 255]
        
        return markers, segmented

    def _calibrate_size(self, image: Image.Image, reference_object: str, preprocessed: Dict) -> Optional[Dict]:
        """Calculate pixel-to-real-world conversion using reference object"""
        if reference_object not in self.reference_sizes:
            return None
            
        ref_size = self.reference_sizes[reference_object]
        ref_real_size = max(ref_size.values())
        edges = preprocessed['edges']
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ref_contour = self._find_reference_contour(contours, reference_object, ref_real_size)
        
        if ref_contour is not None:
            pixel_to_mm = self._calculate_pixel_to_mm(ref_contour, reference_object, ref_real_size)
            return {
                'pixel_to_mm': pixel_to_mm,
                'reference_contour': ref_contour,
                'reference_size_mm': ref_real_size
            }
        return None

    def _find_reference_contour(self, contours: List, reference_object: str, ref_real_size: float) -> Optional[np.ndarray]:
        """Find the contour that best matches the reference object"""
        best_contour = None
        min_diff = float('inf')
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if reference_object in ["Credit Card", "iPhone"]:
                    rect = cv2.minAreaRect(contour)
                    (_, _), (w, h), _ = rect
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    
                    if 1.5 < aspect_ratio < 2.5:
                        pixel_size = max(w, h)
                        diff = abs(pixel_size - ref_real_size)
                        if diff < min_diff:
                            min_diff = diff
                            best_contour = contour
                
                elif reference_object in ["Quarter (US)"]:
                    if 0.85 < circularity < 1.15:
                        pixel_size = np.sqrt(area / np.pi) * 2
                        diff = abs(pixel_size - ref_real_size)
                        if diff < min_diff:
                            min_diff = diff
                            best_contour = contour
        
        return best_contour

    def _calculate_pixel_to_mm(self, contour: np.ndarray, reference_object: str, ref_real_size: float) -> float:
        """Calculate pixel to millimeter ratio"""
        if reference_object in ["Credit Card", "iPhone"]:
            rect = cv2.minAreaRect(contour)
            (_, _), (w, h), _ = rect
            pixel_size = max(w, h)
        else:
            area = cv2.contourArea(contour)
            pixel_size = np.sqrt(area / np.pi) * 2
        
        return ref_real_size / pixel_size

    def _analyze_food_instances(self, preprocessed: Dict, markers: np.ndarray, calibration: Optional[Dict]) -> List[Dict]:
        """Analyze detected food instances"""
        food_instances = []
        hsv = cv2.cvtColor(preprocessed['enhanced'], cv2.COLOR_BGR2HSV)
        
        food_type_ranges = {
            'Meat': [(0, 30, 50), (20, 255, 255)],
            'Vegetables': [(35, 30, 30), (85, 255, 255)],
            'Grains': [(0, 0, 180), (180, 30, 255)]
        }
        
        for marker in range(2, markers.max() + 1):
            marker_mask = (markers == marker).astype(np.uint8)
            contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                instance = self._analyze_food_instance(
                    contours[0], preprocessed, hsv, food_type_ranges, calibration
                )
                if instance:
                    food_instances.append(instance)
        
        return food_instances

    def _analyze_food_instance(
        self, contour: np.ndarray, preprocessed: Dict, hsv: np.ndarray, 
        food_type_ranges: Dict, calibration: Optional[Dict]
    ) -> Optional[Dict]:
        """Analyze individual food instance"""
        area = cv2.contourArea(contour)
        if area < self.settings.MIN_CONTOUR_AREA:
            return None
            
        x, y, w, h = cv2.boundingRect(contour)
        roi = preprocessed['enhanced'][y:y+h, x:x+w]
        roi_hsv = hsv[y:y+h, x:x+w]
        
        food_type = self._determine_food_type(roi_hsv, food_type_ranges)
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        volume_cm3 = None
        if calibration:
            if self.settings.USE_MIDAS_DEPTH and self._depth_model is not None:
                volume_cm3 = self._estimate_volume_midas(roi, contour, calibration['pixel_to_mm'])
            if volume_cm3 is None:
                estimated_height_mm = {'Meat': 20, 'Vegetables': 15, 'Grains': 10}.get(food_type, 15)
                volume_cm3 = self._estimate_volume(contour, estimated_height_mm, calibration['pixel_to_mm'])
        
        return {
            'type': food_type,
            'bbox': (x, y, w, h),
            'area': area,
            'contour': contour,
            'solidity': solidity,
            'volume_cm3': volume_cm3,
            'confidence': self._calculate_confidence(food_type, solidity)
        }

    # ------------------------------------------------------------
    # Depth-based volume estimation helpers
    # ------------------------------------------------------------

    def _compute_depth_map(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return a depth map (H×W) using the MiDaS model; values are arbitrary units."""
        if not self.settings.USE_MIDAS_DEPTH or self._depth_model is None or self._depth_tf is None or torch is None:
            return None
        try:
            input_batch = self._depth_tf(Image.fromarray(img_rgb)).unsqueeze(0)
            if torch.cuda.is_available():
                input_batch = input_batch.to("cuda")
            with torch.no_grad():  # type: ignore[attr-defined]
                prediction = self._depth_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth_map = prediction.cpu().numpy()
                return depth_map
        except Exception as e:
            print(f"[WARN] Depth inference failed: {e}")
            return None

    def _estimate_volume_midas(self, roi_bgr: np.ndarray, contour: np.ndarray, pixel_to_mm: float) -> Optional[float]:
        """Estimate volume using MiDaS depth predictions over the ROI."""
        depth_map = self._compute_depth_map(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        if depth_map is None:
            return None

        # Create mask of contour within ROI
        mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
        # Shift contour coords to ROI local
        x_off, y_off, _, _ = cv2.boundingRect(contour)
        shifted_contour = contour.copy()
        shifted_contour[:, 0, 0] -= x_off
        shifted_contour[:, 0, 1] -= y_off
        cv2.drawContours(mask, [shifted_contour], -1, 255, -1)

        depths = depth_map[mask == 255]
        if depths.size == 0:
            return None
        median_depth = float(np.median(depths))  # type: ignore[arg-type]

        # Heuristic: convert relative depth to approximate mm using scaling constant
        depth_mm = max(median_depth, 1e-3) * 10  # scale factor

        area_pixels = cv2.contourArea(contour)
        area_mm2 = area_pixels * (pixel_to_mm ** 2)
        volume_mm3 = area_mm2 * depth_mm
        return volume_mm3 / 1000.0  # cm^3

    def _determine_food_type(self, roi_hsv: np.ndarray, food_type_ranges: Dict) -> str:
        """Determine food type based on color analysis"""
        max_match = 0
        food_type = 'Unknown'
        
        for type_name, (lower, upper) in food_type_ranges.items():
            mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
            match = cv2.countNonZero(mask) / (roi_hsv.shape[0] * roi_hsv.shape[1])
            if match > max_match:
                max_match = match
                food_type = type_name
        
        return food_type

    def _estimate_volume(self, contour: np.ndarray, height_mm: float, pixel_to_mm: float) -> float:
        """Estimate food volume using contour area and estimated height"""
        area_pixels = cv2.contourArea(contour)
        area_mm2 = area_pixels * (pixel_to_mm ** 2)
        volume_mm3 = area_mm2 * height_mm
        return volume_mm3 / 1000  # Convert to cm³

    def _calculate_confidence(self, food_type: str, solidity: float) -> float:
        """Calculate confidence score for food detection"""
        type_confidence = 0.8 if food_type != 'Unknown' else 0.4
        shape_confidence = min(solidity, 1.0)
        return (type_confidence + shape_confidence) / 2

    def _create_visualization(self, img: np.ndarray, food_instances: List[Dict], calibration: Optional[Dict]) -> np.ndarray:
        """Create visualization of detected food items"""
        result = img.copy()
        
        colors = {
            'Meat': (0, 0, 255),
            'Vegetables': (0, 255, 0),
            'Grains': (255, 0, 0),
            'Unknown': (255, 255, 0)
        }
        
        for instance in food_instances:
            color = colors.get(instance['type'], colors['Unknown'])
            cv2.drawContours(result, [instance['contour']], -1, color, 2)
            
            x, y, _, _ = instance['bbox']
            label = f"{instance['type']}"
            if instance.get('volume_cm3'):
                label += f": {instance['volume_cm3']:.1f}cm³"
            cv2.putText(result, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if calibration and calibration.get('reference_contour') is not None:
            cv2.drawContours(result, [calibration['reference_contour']], -1, (255, 255, 255), 2)
        
        return result 

    # ---------------- YOLO helper ----------------------
    def _yolo_results_to_instances(self, results) -> List[Dict]:
        """Convert ultralytics Results object to our food_instances schema."""
        instances: List[Dict] = []
        if not results:
            return instances
        res = results[0]
        names = res.names  # class id to name
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, 'food')
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            instances.append({
                'type': cls_name,
                'bbox': xyxy.tolist(),
                'confidence': conf,
                'volume_cm3': None  # will require depth estimation later
            })
        return instances 

    def _mask_results_to_instances(self, outputs) -> List[Dict]:
        instances: List[Dict] = []
        if outputs is None:
            return instances
        def to_numpy(arr):
            if hasattr(arr, 'cpu'):
                return arr.cpu().numpy()
            return np.array(arr)

        scores = to_numpy(outputs['scores'])
        labels = to_numpy(outputs['labels'])
        boxes = to_numpy(outputs['boxes'])
        masks = to_numpy(outputs['masks']) if 'masks' in outputs else None
        for i, score in enumerate(scores):
            if score < self.settings.MASK_RCNN_MIN_SCORE:
                continue
            label_id = int(labels[i])
            cls_name = f'class_{label_id}'
            instances.append({
                'type': cls_name,
                'bbox': boxes[i].tolist(),
                'confidence': float(score),
                'mask': masks[i][0].tolist() if masks is not None else None,
                'volume_cm3': None
            })
        return instances 