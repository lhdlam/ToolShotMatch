import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional, Dict
from scipy.spatial import KDTree
from webcolors import CSS3_NAMES_TO_HEX, hex_to_rgb
import concurrent.futures
from functools import lru_cache
import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image as image_processing
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import os
from color_shade import RED, BLUE

class ImageAnalyzer:
    """
    A class for analyzing and comparing images, specifically designed for:
    - Detecting dominant colors in images
    - Comparing similarity between input and reference images
    - Real-time processing with high accuracy
    
    """
    
    def __init__(self, image_ref1: Union[str, Path] = '', image_ref2: Union[str, Path] = ''):
        """
        Initialize the ImageAnalyzer with configuration parameters.
        
        Args:
            min_similarity (float): Minimum similarity threshold (0-1).
                                  Default is 0.95 for high accuracy requirements.
        """
        self.image_ref1 = image_ref1
        self.image_ref2 = image_ref2
        vgg16 = VGG16(weights='imagenet', include_top=False, 
              pooling='max', input_shape=(32, 32, 3))
        self.model = vgg16
        for model_layer in self.model.layers:
            model_layer.trainable = False

        # Khởi tạo color mapping cho việc chuyển đổi RGB sang tên màu
        self._initialize_color_mapping()
        # Định nghĩa ngưỡng cho pixel trắng trong không gian HSV
        self.white_threshold = {
            'saturation_max': 30,  # Ngưỡng độ bão hòa tối đa cho màu trắng
            'value_min': 200       # Ngưỡng giá trị tối thiểu cho màu trắng
        }
        # Pre-compute thresholds for better performance
        self._compute_thresholds()

    def _compute_thresholds(self):
        """
        Pre-compute thresholds và masks cho việc tối ưu hóa
        """
        # Tạo mask cho việc phát hiện màu trắng trong HSV space
        s_max = self.white_threshold['saturation_max']
        v_min = self.white_threshold['value_min']
        
        # Pre-compute ranges cho việc phát hiện màu trắng
        self.white_lower = np.array([0, 0, v_min])
        self.white_upper = np.array([180, s_max, 255])

    @lru_cache(maxsize=128)
    def _get_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """
        Cached version của hàm chuyển đổi RGB sang tên màu.
        
        Args:
            rgb_color: Tuple RGB (r, g, b)
            
        Returns:
            str: Tên màu tiếng Anh
        """
        distance, index = self.color_tree.query(rgb_color)
        return self.color_names[index]
    
    def get_dominant_color(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Optimized version của hàm lấy màu chủ đạo.
        
        Args:
            image: Ảnh BGR numpy array
            
        Returns:
            Tuple[np.ndarray, str]: (Giá trị màu BGR, tên màu)
        """
        # Chuyển sang HSV và tìm pixels không phải màu trắng
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        non_white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        non_white_mask = cv2.bitwise_not(non_white_mask)
        
        if cv2.countNonZero(non_white_mask) == 0:
            raise ValueError("No dominant color found - image appears to be all white")
            
        # Tính trung bình màu của các pixel không phải màu trắng
        dominant_color = cv2.mean(image, mask=non_white_mask)[:3]
        dominant_color = np.uint8([[dominant_color]])
        
        # Chuyển sang RGB để lấy tên màu
        rgb_color = cv2.cvtColor(dominant_color, cv2.COLOR_BGR2RGB)[0][0]
        color_name = self._get_color_name(tuple(rgb_color))
        
        return dominant_color[0], color_name


    # Fix the color mapping initialization
    def _initialize_color_mapping(self) -> None:
        """
        Initialize color mapping to convert RGB to color names.
        Uses KDTree for efficient nearest color lookup.
        """
        # Convert CSS3 colors to RGB
        rgb_colors = []
        self.color_names = []
        
        for name, hex_color in CSS3_NAMES_TO_HEX.items():
            rgb = hex_to_rgb(hex_color)
            rgb_colors.append(rgb)
            self.color_names.append(name)
            
        # Create KDTree for nearest color lookup
        self.color_tree = KDTree(rgb_colors)
        
    
    def _is_white_pixel(self, pixel: np.ndarray) -> bool:
        """
        Kiểm tra xem một pixel HSV có phải là màu trắng không.
        
        Args:
            pixel: Mảng numpy chứa giá trị HSV của pixel
            
        Returns:
            bool: True nếu pixel được coi là màu trắng
        """
        # H: Hue, S: Saturation, V: Value
        h, s, v = pixel
        return s <= self.white_threshold['saturation_max'] and v >= self.white_threshold['value_min']
    
    def get_image_embeddings(self, object_image : image_processing):
    
        """
        -----------------------------------------------------
        convert image into 3d array and add additional dimension for model input
        -----------------------------------------------------
        return embeddings of the given image
        """

        image_array = np.expand_dims(image_processing.img_to_array(object_image), axis = 0)
        image_embedding = self.model.predict(image_array)

        return image_embedding
    
    def compare_images(self, 
                      input_image: Union[str, Path],
                      reference_image: Union[str, Path]
                     ) -> float:
        """
        So sánh hai ảnh sử dụng nhiều phương pháp để đạt độ chính xác cao nhất.
        
        Args:
            input_image: Ảnh input dạng BGR
            reference_image: Ảnh tham khảo dạng BGR
            
        Returns:
            float: Điểm tương đồng (0-1)
        """
        # Validate kích thước ảnh
        # self._validate_images_size(input_image, reference_image)

        input_image = Image.open(input_image).convert("RGB").resize((32, 32))
        ref_image = Image.open(reference_image).convert("RGB").resize((32, 32))
        
        # Chuyển ảnh sang grayscale cho việc so sánh cấu trúc
        first_image_vector = self.get_image_embeddings(input_image)
        second_image_vector = self.get_image_embeddings(ref_image)
        
        similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
        
        return similarity_score[0]

    def _validate_image_path(self, image_path: Union[str, Path]) -> bool:
        """
        Validate if the image path exists and has a supported format.
        
        Args:
            image_path: Path to the image file (string or Path object)
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            ValueError: If image format is not supported
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
            
        Raises:
            ValueError: If image cannot be loaded
        """
        self._validate_image_path(image_path)
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        return image
    
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
            image = future_input.result()
            
            # Input validation
            if margin_size < 0:
                raise ValueError("Margin size must be positive")
            if margin_size * 2 >= min(image.shape[0], image.shape[1]):
                raise ValueError("Margin size too large for image dimensions")
                
            # Calculate new dimensions
            height, width = image.shape[:2]
            new_height = height - (2 * margin_size)
            new_width = width - (2 * margin_size)
            
            # Crop image using array slicing
            cropped = image[margin_size:height-margin_size, 
                        margin_size:width-margin_size]
            
            future_input_color = executor.submit(self.get_dominant_color, cropped)
            _, dominant_input_color_name = future_input_color.result()
            dominant_input_color_name = self.normalize_color(dominant_input_color_name, RED, BLUE)
            if dominant_input_color_name == "red":
                return int(0)
            else:
                return int(1)


    def process(self, 
                input_path: Union[str, Path], 
               ) -> Tuple[str, float]:
        """
        Optimized version của hàm xử lý chính.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Đọc ảnh song song
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
            
            print(f"Input image dominant color: {dominant_input_color_name}")
            print(f"Reference image 1 dominant color: {dominant_ref1_color_name}")
            print(f"Reference image 2 dominant color: {dominant_ref2_color_name}")

            dominant_input_color_name = self.normalize_color(dominant_input_color_name, RED, "red")
            dominant_ref1_color_name = self.normalize_color(dominant_ref1_color_name, RED, "red")
            dominant_ref2_color_name = self.normalize_color(dominant_ref2_color_name, RED, "red")

            dominant_input_color_name = self.normalize_color(dominant_input_color_name, BLUE, "blue")
            dominant_ref1_color_name = self.normalize_color(dominant_ref1_color_name, BLUE, "blue")
            dominant_ref2_color_name = self.normalize_color(dominant_ref2_color_name, BLUE, "blue")

            basic_colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray"]
            for color in basic_colors:
                if color in dominant_input_color_name:
                    dominant_input_color_name = color
                if color in dominant_ref1_color_name:
                    dominant_ref1_color_name = color
                if color in dominant_ref2_color_name:
                    dominant_ref2_color_name = color

            if dominant_input_color_name == dominant_ref1_color_name:
                reference_image = self.image_ref1
            elif dominant_input_color_name == dominant_ref2_color_name:
                reference_image = self.image_ref2
            else:
                raise ValueError("Input image color does not match any reference image color")


            future_similarity = executor.submit(
                self.compare_images, input_path, reference_image
            )
            
            similarity = future_similarity.result()
            
        similarity_percentage = similarity * 100
        return dominant_input_color_name, similarity_percentage