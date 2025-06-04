from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..config.settings import Settings
import io

class ImageProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.reference_sizes = {
            "Credit Card": {"width": 85.60, "height": 53.98},
            "Quarter (US)": {"diameter": 24.26},
            "Standard Plate": {"diameter": 250},
            "iPhone": {"width": 71.5, "height": 146.7}
        }

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