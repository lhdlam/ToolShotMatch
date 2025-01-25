import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, List
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
from skimage.metrics import structural_similarity as ssim
import concurrent.futures
from functools import lru_cache
from color_shade import RED, BLUE

class ImageAnalyzerv2:
    def __init__(self, min_similarity: float = 0.95, cache_size: int = 128, image_ref1: Union[str, Path] = '', image_ref2: Union[str, Path] = ''):
        """
        Initialize the ImageAnalyzer with configuration parameters.
        
        Args:
            min_similarity (float): Minimum similarity threshold (0-1)
            cache_size (int): Size of LRU cache for color names
        """
        self.image_ref1 = image_ref1
        self.image_ref2 = image_ref2

        self.min_similarity = min_similarity
        self.cache_size = cache_size
        self._initialize_color_mapping()
        self.white_threshold = {
            'saturation_max': 30,
            'value_min': 200
        }
        self._compute_thresholds()
        
    def _initialize_color_mapping(self):
        """
        Initialize color mapping for RGB to color name conversion.
        """
        rgb_colors = []
        self.color_names = []
        
        for hex_color, name in CSS3_HEX_TO_NAMES.items():
            rgb = hex_to_rgb(hex_color)
            rgb_colors.append(rgb)
            self.color_names.append(name)
            
        self.color_tree = KDTree(rgb_colors)
        
    def _compute_thresholds(self):
        """
        Pre-compute thresholds for white color detection.
        """
        s_max = self.white_threshold['saturation_max']
        v_min = self.white_threshold['value_min']
        
        self.white_lower = np.array([0, 0, v_min])
        self.white_upper = np.array([180, s_max, 255])
        
    @lru_cache(maxsize=128)
    def _get_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """
        Cached method to convert RGB to color name.
        
        Args:
            rgb_color: Tuple RGB (r, g, b)
            
        Returns:
            str: Color name in English
        """
        distance, index = self.color_tree.query(rgb_color)
        return self.color_names[index]
        
    def _validate_image_path(self, image_path: Union[str, Path]) -> bool:
        """
        Validate if the image path exists and has a supported format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If format is not supported
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        if path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats are: {supported_formats}"
            )
            
        return True
        
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and validate an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Loaded image in BGR format
        """
        self._validate_image_path(image_path)
        
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (32,32))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        return image
        
    def _validate_images_size(self, image1: np.ndarray, image2: np.ndarray) -> bool:
        """
        Validate if two images have the same dimensions.
        
        Args:
            image1: First image array
            image2: Second image array
            
        Returns:
            bool: True if images have same dimensions
            
        Raises:
            ValueError: If images have different dimensions
        """
        if image1.shape != image2.shape:
            raise ValueError(
                f"Images have different dimensions: "
                f"{image1.shape} vs {image2.shape}"
            )
        return True
        
    def get_dominant_color(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Get the dominant non-white color from an image.
        
        Args:
            image: BGR numpy array
            
        Returns:
            Tuple[np.ndarray, str]: (BGR color values, color name)
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        non_white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        non_white_mask = cv2.bitwise_not(non_white_mask)
        
        if cv2.countNonZero(non_white_mask) == 0:
            raise ValueError("No dominant color found - image appears to be all white")
            
        dominant_color = cv2.mean(image, mask=non_white_mask)[:3]
        dominant_color = np.uint8([[dominant_color]])
        
        rgb_color = cv2.cvtColor(dominant_color, cv2.COLOR_BGR2RGB)[0][0]
        color_name = self._get_color_name(tuple(rgb_color))
        
        return dominant_color[0], color_name
        
    def _parallel_histogram_comparison(self,
                                    hsv1: np.ndarray,
                                    hsv2: np.ndarray,
                                    channel: int,
                                    bins: int = 32) -> float:
        """
        Compare histogram for a single color channel.
        
        Args:
            hsv1, hsv2: HSV images
            channel: Color channel (0=H, 1=S, 2=V)
            bins: Number of bins
            
        Returns:
            float: Histogram similarity score
        """
        hist1 = cv2.calcHist([hsv1], [channel], None, [bins], [0, 256])
        hist2 = cv2.calcHist([hsv2], [channel], None, [bins], [0, 256])
        
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
    def _compare_color_histograms(self,
                                img1: np.ndarray,
                                img2: np.ndarray,
                                bins: int = 32) -> float:
        """
        Compare color histograms in parallel.
        
        Args:
            img1, img2: Input images
            bins: Number of histogram bins
            
        Returns:
            float: Overall histogram similarity
        """
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for channel in range(3):
                future = executor.submit(
                    self._parallel_histogram_comparison,
                    hsv1, hsv2, channel, bins
                )
                futures.append(future)
                
            hist_scores = [future.result() for future in futures]
            
        return np.mean(hist_scores)
        
    def compare_images(self, 
                      input_image: np.ndarray, 
                      reference_image: np.ndarray
                     ) -> float:
        """
        Compare two images using multiple methods.
        
        Args:
            input_image: Input BGR image
            reference_image: Reference BGR image
            
        Returns:
            float: Similarity score (0-1)
        """
        self._validate_images_size(input_image, reference_image)
        
        gray1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        
        ssim_score, _ = ssim(gray1, gray2, full=True)
        hist_score = self._compare_color_histograms(input_image, reference_image)
        
        weights = {
            'ssim': 0.4,
            'histogram': 0.6
        }
        
        final_score = (
            ssim_score * weights['ssim'] +
            hist_score * weights['histogram']
        )
        
        return final_score

    def normalize_color(self, color_name, color_group_red, color_group_blue):
        if "red" in color_name:
            return "red"
        elif "blue" in color_name:
            return "blue"
        elif color_name in color_group_red:
            return "red"
        elif color_name in color_group_blue:
            return "blue"

    def booling_cursor(self, image_path: Union[str, Path], margin_size):
        """
        Crop equal margins from all edges of the image
        
        Args:
            image: numpy array or PIL Image
            margin_size: number of pixels to crop from each edge
            
        Returns:
            Cropped image as numpy array
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Đọc ảnh song song
            future_input = executor.submit(self._load_image, image_path)
            future_reference1 = executor.submit(self._load_image, self.image_ref1)
            future_reference2 = executor.submit(self._load_image, self.image_ref2)

            input_image = future_input.result()
            reference_image1 = future_reference1.result()
            reference_image2 = future_reference2.result()
            
            # Input validation
            if margin_size < 0:
                raise ValueError("Margin size must be positive")
            if margin_size * 2 >= min(input_image.shape[0], input_image.shape[1]):
                raise ValueError("Margin size too large for image dimensions")
                
            # Calculate new dimensions
            height, width = input_image.shape[:2]
            new_height = height - (2 * margin_size)
            new_width = width - (2 * margin_size)
            
            # Crop image using array slicing
            cropped = input_image[margin_size:height-margin_size, 
                        margin_size:width-margin_size]
            
            # Xử lý song song màu chủ đạo và so sánh ảnh
            future_input_color = executor.submit(self.get_dominant_color, input_image)
            future_ref1_color = executor.submit(self.get_dominant_color, reference_image1)
            future_ref2_color = executor.submit(self.get_dominant_color, reference_image2)

            _, dominant_input_color_name = future_input_color.result()
            _, dominant_ref1_color_name = future_ref1_color.result()
            _, dominant_ref2_color_name = future_ref2_color.result()

            dominant_input_color_name = self.normalize_color(dominant_input_color_name, RED, BLUE)
            dominant_ref1_color_name = self.normalize_color(dominant_ref1_color_name, RED, BLUE)
            dominant_ref2_color_name = self.normalize_color(dominant_ref2_color_name, RED, BLUE)

            print(f"Input color: {dominant_input_color_name}")
            print(f"Reference 1 color: {dominant_ref1_color_name}")
            print(f"Reference 2 color: {dominant_ref2_color_name}")

            if dominant_input_color_name == dominant_ref1_color_name:
                reference_image = reference_image1
            elif dominant_input_color_name == dominant_ref2_color_name:
                reference_image = reference_image2
            else:
                return int(2)
            
            future_similarity = executor.submit(
                self.compare_images, input_image, reference_image
            )
            
            similarity = future_similarity.result()
            print(f"Similarity: {similarity}")
            
            if similarity <= self.min_similarity:
                return int(2)
            elif similarity > self.min_similarity and dominant_input_color_name == "red":
                return int(0)
            elif similarity > self.min_similarity and dominant_input_color_name == "blue":
                return int(1)
        
    def process(self, 
                input_path: Union[str, Path]
               ) -> Tuple[str, float]:
        """
        Main processing method to analyze and compare images.
        
        Args:
            input_path: Path to input image
            reference_path: Path to reference image
            
        Returns:
            Tuple[str, float]: (dominant color name, similarity percentage)
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_input = executor.submit(self._load_image, input_path)
            future_reference1 = executor.submit(self._load_image, self.image_ref1)
            future_reference2 = executor.submit(self._load_image, self.image_ref2)

            input_image = future_input.result()
            reference_image1 = future_reference1.result()
            reference_image2 = future_reference2.result()
            
            
            # Xử lý song song màu chủ đạo và so sánh ảnh
            future_input_color = executor.submit(self.get_dominant_color, input_image)
            future_ref1_color = executor.submit(self.get_dominant_color, reference_image1)
            future_ref2_color = executor.submit(self.get_dominant_color, reference_image2)

            _, dominant_input_color_name = future_input_color.result()
            _, dominant_ref1_color_name = future_ref1_color.result()
            _, dominant_ref2_color_name = future_ref2_color.result()

            dominant_input_color_name = self.normalize_color(dominant_input_color_name, RED, BLUE)
            dominant_ref1_color_name = self.normalize_color(dominant_ref1_color_name, RED, BLUE)
            dominant_ref2_color_name = self.normalize_color(dominant_ref2_color_name, RED, BLUE)

            if dominant_input_color_name == dominant_ref1_color_name:
                reference_image = reference_image1
            elif dominant_input_color_name == dominant_ref2_color_name:
                reference_image = reference_image2
            else:
                raise ValueError("Input image color does not match any reference image color")

            future_similarity = executor.submit(
                self.compare_images, input_image, reference_image
            )
            
            similarity = future_similarity.result()
            
        similarity_percentage = similarity * 100

        return dominant_input_color_name, similarity_percentage