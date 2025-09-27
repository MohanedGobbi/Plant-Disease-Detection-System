import sys
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import tempfile
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from utils.similarity import SymptomMatcher
import json

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QPushButton, QLabel, QTextEdit, QSlider, QComboBox,
                             QFileDialog, QGroupBox, QScrollArea, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtMultimedia import QCameraInfo, QCamera, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder

# ----------------------------
# Device Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Text Preprocessing
# ----------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and tokenize text for BiLSTM model"""
    text = re.sub(r"[^a-z\s]", "", text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return tokens

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image_for_prediction(image):
    """Preprocess image for ResNet18 feature extraction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.open(image).convert("RGB")
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# ----------------------------
# Image Filters
# ----------------------------
def apply_grayscale(image):
    """Convert image to grayscale"""
    if image.mode != 'L':
        return image.convert("L")
    return image

def apply_contrast(image, factor):
    """Adjust image contrast"""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_brightness(image, factor):
    """Adjust image brightness"""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_blur(image, radius):
    """Apply Gaussian blur to image"""
    from PIL import ImageFilter
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_sharpness(image, factor):
    """Adjust image sharpness"""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def apply_saturation(image, factor):
    """Adjust image saturation"""
    from PIL import ImageEnhance
    if image.mode == 'L':
        image = image.convert('RGB')
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def apply_warm_filter(image):
    """Apply warm color filter"""
    if image.mode == 'L':
        image = image.convert('RGB')
    data = np.array(image)
    data[:, :, 0] = np.clip(data[:, :, 0] + 20, 0, 255)  # Increase red
    data[:, :, 2] = np.clip(data[:, :, 2] - 10, 0, 255)  # Decrease blue
    return Image.fromarray(data)

def apply_cool_filter(image):
    """Apply cool color filter"""
    if image.mode == 'L':
        image = image.convert('RGB')
    data = np.array(image)
    data[:, :, 2] = np.clip(data[:, :, 2] + 20, 0, 255)  # Increase blue
    data[:, :, 0] = np.clip(data[:, :, 0] - 10, 0, 255)  # Decrease red
    return Image.fromarray(data)

def apply_sepia(image):
    """Apply sepia filter to image"""
    if image.mode == 'L':
        image = image.convert('RGB')
    data = np.array(image)
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_data = np.dot(data, sepia_matrix.T)
    sepia_data = np.clip(sepia_data, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_data)

# ----------------------------
# Load Models
# ----------------------------
# Load XGBoost model
xgb_clf = joblib.load("models/xgb_plant_disease.pkl")

# Load ResNet18 feature extractor
resnet_feat = torch.load("models/resnet18_feature_extractor_full.pth", 
                         map_location=device, weights_only=False)
resnet_feat.to(device)
resnet_feat.eval()

# Load BiLSTM components
vocab = joblib.load("models/vocab.pkl")
le = joblib.load("models/label_encoder.pkl")
embedding_matrix = np.load("models/embedding_matrix.npy")

# Load BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes, embedding_matrix=None):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=False,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(torch.cat((h[-2], h[-1]), 1))

model = BiLSTM(len(vocab)+1, 100, 128, len(le.classes_), embedding_matrix)
model.load_state_dict(torch.load("models/bilstm_model.pth", map_location=device, weights_only=False))
model.to(device)
model.eval()

# ----------------------------
# Prediction Functions
# ----------------------------
def predict_image(image_input):
    """Predict plant disease from image and return label with probability"""
    image_tensor = preprocess_image_for_prediction(image_input).to(device)
    
    with torch.no_grad():
        feat = resnet_feat(image_tensor)
        feat = feat.view(feat.size(0), -1).cpu().numpy()
    
    pred = xgb_clf.predict(feat)[0]
    proba = xgb_clf.predict_proba(feat)[0]
    
    label = "Diseased" if pred == 0 else "Healthy"
    confidence = max(proba) * 100  # Convert to percentage
    
    return label, confidence

def predict_text(text):
    """Predict plant disease from text description and return label with probability"""
    tokens = clean_text(text)
    indices = [vocab.get(t, 0) for t in tokens]
    
    if not indices:
        return "Unknown", 0.0
    
    X = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][pred_idx].item() * 100  # Convert to percentage
    
    result = le.inverse_transform([pred_idx])[0]
    return result, confidence

# ----------------------------
# Main Application Window
# ----------------------------
class PlantDiseaseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.original_image = None
        self.camera = None
        self.image_capture = None
        
        # Initialize symptom matcher
        self.symptom_matcher = SymptomMatcher("data/plantwild_prompts.json")
        
        self.initUI()
        self.load_treatment_info()
        
    def initUI(self):
        self.setWindowTitle('Plant Disease Detection System')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Tab widget for different input methods
        self.tabs = QTabWidget()
        
        # Setup tabs
        self.setup_upload_tab()
        self.setup_camera_tab()
        self.setup_text_tab()
        self.setup_similarity_tab()
        
        left_layout.addWidget(self.tabs)
        
        # Image filters section
        filters_group = QGroupBox("Image Filters (Display Only)")
        filters_layout = QVBoxLayout()
        
        filter_label = QLabel("Select Filter:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Grayscale", "Contrast", "Brightness", "Blur", 
                                   "Sharpness", "Saturation", "Warm", "Cool", "Sepia"])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        
        self.intensity_label = QLabel("Intensity: 1.0")
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(1, 20)
        self.intensity_slider.setValue(10)
        self.intensity_slider.valueChanged.connect(self.update_intensity_label)
        self.intensity_slider.valueChanged.connect(self.apply_filter)
        
        filters_layout.addWidget(filter_label)
        filters_layout.addWidget(self.filter_combo)
        filters_layout.addWidget(self.intensity_label)
        filters_layout.addWidget(self.intensity_slider)
        filters_group.setLayout(filters_layout)
        
        left_layout.addWidget(filters_group)
        
        # Predict button
        self.predict_btn = QPushButton("Predict Disease")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; }")
        left_layout.addWidget(self.predict_btn)
        
        # Right panel for output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setText("Image will appear here")
        right_layout.addWidget(self.image_label)
        
        # Results display
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout()
        
        self.result_label = QLabel("No prediction yet")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 16px; padding: 10px; font-weight: bold;")
        
        self.confidence_label = QLabel("")
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setStyleSheet("font-size: 14px; color: #666;")
        
        # Add progress bar for confidence visualization
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setVisible(False)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        
        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.confidence_label)
        results_layout.addWidget(self.confidence_bar)
        results_group.setLayout(results_layout)
        
        right_layout.addWidget(results_group)
        
        # Treatment information
        treatment_group = QGroupBox("Treatment Suggestions")
        treatment_layout = QVBoxLayout()
        
        self.treatment_text = QTextEdit()
        self.treatment_text.setReadOnly(True)
        self.treatment_text.setStyleSheet("font-size: 14px; line-height: 1.4;")
        
        treatment_layout.addWidget(self.treatment_text)
        treatment_group.setLayout(treatment_layout)
        
        right_layout.addWidget(treatment_group)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
    def setup_upload_tab(self):
        self.upload_tab = QWidget()
        layout = QVBoxLayout(self.upload_tab)
        
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setStyleSheet("QPushButton { padding: 10px; }")
        
        layout.addWidget(self.upload_btn)
        layout.addStretch()
        self.tabs.addTab(self.upload_tab, "Upload Image")
    
    def setup_camera_tab(self):
        self.camera_tab = QWidget()
        layout = QVBoxLayout(self.camera_tab)
        
        # Camera viewfinder
        self.camera_viewfinder = QCameraViewfinder()
        layout.addWidget(self.camera_viewfinder)
        
        # Camera controls
        controls_layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        cameras = QCameraInfo.availableCameras()
        for camera in cameras:
            self.camera_combo.addItem(camera.description())
        
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        
        self.capture_btn = QPushButton("Capture Photo")
        self.capture_btn.clicked.connect(self.capture_photo)
        self.capture_btn.setEnabled(False)
        
        controls_layout.addWidget(self.camera_combo)
        controls_layout.addWidget(self.start_camera_btn)
        controls_layout.addWidget(self.capture_btn)
        
        layout.addLayout(controls_layout)
        layout.addStretch()
        self.tabs.addTab(self.camera_tab, "Take Photo")
    
    def setup_text_tab(self):
        self.text_tab = QWidget()
        layout = QVBoxLayout(self.text_tab)
        
        instruction = QLabel("Describe your plant's symptoms:")
        instruction.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("e.g., My tomato plant has yellow leaves with black spots...")
        
        layout.addWidget(instruction)
        layout.addWidget(self.text_input)
        layout.addStretch()
        self.tabs.addTab(self.text_tab, "Text Description")
    
    def setup_similarity_tab(self):
        self.similarity_tab = QWidget()
        layout = QVBoxLayout(self.similarity_tab)
    
        # Method selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Matching Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Both", "TF-IDF", "Sentence Transformer"])
    
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)
    
        # Query input
        instruction = QLabel("Describe your plant's symptoms for similarity matching:")
        instruction.setStyleSheet("font-size: 14px; font-weight: bold;")
    
        self.similarity_input = QTextEdit()
        self.similarity_input.setPlaceholderText("e.g., yellow spots on leaves with black centers...")
    
        layout.addWidget(instruction)
        layout.addWidget(self.similarity_input)
    
        # Match button
        self.match_btn = QPushButton("Find Similar Diseases")
        self.match_btn.clicked.connect(self.match_symptoms)
        self.match_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 10px; }")
        layout.addWidget(self.match_btn)
    
        # Results area
        self.similarity_results = QTextEdit()
        self.similarity_results.setReadOnly(True)
        self.similarity_results.setStyleSheet("font-size: 14px; line-height: 1.4;")
    
        layout.addWidget(self.similarity_results)
        layout.addStretch()
        self.tabs.addTab(self.similarity_tab, "Symptom Matching")

    def match_symptoms(self):
        query = self.similarity_input.toPlainText().strip()
        if not query:
            self.similarity_results.setText("Please enter a description of symptoms.")
            return
    
        method = self.method_combo.currentText().lower().replace(" ", "")
        if method == "both":
            method = "both"
    
        try:
            results = self.symptom_matcher.match_symptoms(query, method=method, top_k=5)
        
            if not results:
                self.similarity_results.setText("No matching diseases found. Try a more detailed description.")
                return
            
            result_text = "🔍 Top Matching Diseases:\n\n"
            for i, result in enumerate(results, 1):
                result_text += f"{i}. {result['disease'].title()}\n"
                result_text += f"   Similarity: {result['similarity']:.3f} ({result['method']})\n"
                result_text += f"   Description: {result['description']}\n\n"
        
            self.similarity_results.setText(result_text)
        
        except Exception as e:
            self.similarity_results.setText(f"Error during symptom matching: {str(e)}")

    def start_camera(self):
        camera_index = self.camera_combo.currentIndex()
        cameras = QCameraInfo.availableCameras()
        
        if cameras:
            self.camera = QCamera(cameras[camera_index])
            self.image_capture = QCameraImageCapture(self.camera)
            
            self.camera.setViewfinder(self.camera_viewfinder)
            self.camera.start()
            
            self.capture_btn.setEnabled(True)
            self.start_camera_btn.setEnabled(False)
    
    def capture_photo(self):
        if self.image_capture:
            self.image_capture.capture()
            self.image_capture.imageCaptured.connect(self.on_image_captured)
    
    def on_image_captured(self, id, qimage):
        # Convert QImage to PIL Image
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)
        
        self.original_image = Image.fromarray(arr, 'RGB')
        self.current_image = self.original_image.copy()
        self.display_image(self.current_image)
        
        if self.camera:
            self.camera.stop()
        self.start_camera_btn.setEnabled(True)
        self.capture_btn.setEnabled(False)
    
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.original_image = Image.open(file_path).convert("RGB")
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
    
    def display_image(self, image):
        # Convert PIL image to QPixmap
        if image.mode == 'RGB':
            data = image.tobytes('raw', 'RGB')
            qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
        elif image.mode == 'L':  # Grayscale
            data = image.tobytes('raw', 'L')
            qimage = QImage(data, image.width, image.height, QImage.Format_Grayscale8)
        elif image.mode == 'RGBA':
            data = image.tobytes('raw', 'RGBA')
            qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        else:
            # Convert to RGB if unknown format
            image = image.convert('RGB')
            data = image.tobytes('raw', 'RGB')
            qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def update_intensity_label(self):
        value = self.intensity_slider.value()
        self.intensity_label.setText(f"Intensity: {value/10:.1f}")
    
    def apply_filter(self):
        if self.original_image is None:
            return
        
        filter_type = self.filter_combo.currentText()
        intensity = self.intensity_slider.value() / 10
        
        # Always apply filters to a copy of the original image
        image = self.original_image.copy()
        
        try:
            if filter_type == "Grayscale":
                image = apply_grayscale(image)
            elif filter_type == "Contrast":
                image = apply_contrast(image, intensity)
            elif filter_type == "Brightness":
                image = apply_brightness(image, intensity)
            elif filter_type == "Blur":
                image = apply_blur(image, intensity)
            elif filter_type == "Sharpness":
                image = apply_sharpness(image, intensity)
            elif filter_type == "Saturation":
                image = apply_saturation(image, intensity)
            elif filter_type == "Warm":
                image = apply_warm_filter(image)
            elif filter_type == "Cool":
                image = apply_cool_filter(image)
            elif filter_type == "Sepia":
                image = apply_sepia(image)
            
            self.current_image = image
            self.display_image(image)
        except Exception as e:
            # Reset to original image if filter fails
            self.current_image = self.original_image.copy()
            self.display_image(self.original_image)
    
    def save_temp_image(self, image):
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image.save(temp_file.name, 'JPEG')
        return temp_file.name
    
    def predict(self):
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 2:  # Text tab
            text = self.text_input.toPlainText().strip()
            if text:
                label, confidence = predict_text(text)
                self.display_results(label, confidence, "text")
            else:
                self.reset_results()
                self.result_label.setText("Please enter a description of your plant's symptoms.")
        elif current_tab == 3:  # Symptom matching tab
            self.reset_results()
            self.result_label.setText("Use the 'Find Similar Diseases' button for symptom matching.")
        else:  # Image tabs
            if self.original_image:
                # Use original image for prediction, not filtered one
                temp_path = self.save_temp_image(self.original_image)
                label, confidence = predict_image(temp_path)
                self.display_results(label, confidence, "image")
                os.unlink(temp_path)
            else:
                self.reset_results()
                self.result_label.setText("Please upload or capture an image first.")
    
    def reset_results(self):
        """Reset all result displays"""
        self.confidence_label.setText("")
        self.confidence_bar.setVisible(False)
        self.treatment_text.setText("")
    
    def display_results(self, label, confidence, input_type):
        # Display prediction result
        self.result_label.setText(f"Prediction: {label}")
        
        # Display confidence with color coding
        if confidence >= 80:
            color = "#4CAF50"  # Green for high confidence
            confidence_text = "High"
        elif confidence >= 60:
            color = "#FF9800"  # Orange for medium confidence
            confidence_text = "Medium"
        else:
            color = "#F44336"  # Red for low confidence
            confidence_text = "Low"
        
        self.confidence_label.setText(
            f"Confidence: {confidence:.1f}% ({confidence_text})"
        )
        self.confidence_label.setStyleSheet(f"font-size: 14px; color: {color}; font-weight: bold;")
        
        # Show confidence bar
        self.confidence_bar.setValue(int(confidence))
        self.confidence_bar.setVisible(True)
        
        # Display input type info
        if input_type == "image":
            method_info = "Image-based prediction using ResNet18 + XGBoost"
        else:
            method_info = "Text-based prediction using BiLSTM"
        
        # Show treatment information
        self.show_treatment(label.lower())
    
    def show_treatment(self, label_lower):
        """Show appropriate treatment based on prediction"""
        if "healthy" in label_lower:
            self.treatment_text.setText(self.treatment_info["healthy"])
            return
        
        # Check for specific diseases
        specific_diseases = {
            "powdery mildew": "powdery" in label_lower or "mildew" in label_lower,
            "leaf spot": "spot" in label_lower,
            "blight": "blight" in label_lower,
            "rust": "rust" in label_lower,
            "root rot": "rot" in label_lower,
            "coffee leaf": "coffee" in label_lower and "leaf" in label_lower,
            "apple rust": "apple" in label_lower and "rust" in label_lower
        }
        
        for disease, is_present in specific_diseases.items():
            if is_present and disease in self.treatment_info:
                self.treatment_text.setText(self.treatment_info[disease])
                return
        
        # Default to generic diseased treatment
        self.treatment_text.setText(self.treatment_info["diseased"])
    
    def load_treatment_info(self):
        """Load treatment information for different diseases"""
        self.treatment_info = {
            "healthy": """✅ Your plant appears healthy! Continue with your current care routine.

General maintenance tips:
• Water appropriately for your plant species
• Ensure adequate sunlight
• Use proper soil and fertilizer
• Monitor for pests regularly
• Maintain good air circulation""",
            
            "diseased": """⚠️ Your plant shows signs of disease. Recommended actions:

Immediate steps:
1. Isolate the plant to prevent spread
2. Remove affected leaves or parts
3. Apply appropriate fungicide or treatment
4. Improve air circulation around the plant
5. Avoid overhead watering to prevent moisture on leaves
6. Consider soil testing for nutrient deficiencies

Consult a local plant specialist for specific treatment recommendations.""",
            
            "powdery mildew": """🍄 Treatment for Powdery Mildew:

Organic treatment:
1. Mix 1 tablespoon baking soda with 1 gallon of water
2. Add a few drops of dish soap as a spreader
3. Spray on affected plants every 7-10 days
4. Improve air circulation around plants
5. Avoid overhead watering
6. Apply sulfur-based fungicide as preventive measure""",
            
            "leaf spot": """🍃 Treatment for Leaf Spot:

Treatment steps:
1. Remove and destroy affected leaves immediately
2. Apply copper-based fungicide
3. Avoid overhead watering
4. Water early in the day so leaves dry quickly
5. Space plants properly for good air circulation
6. Clean up plant debris in fall to prevent overwintering""",
            
            "blight": """💀 Treatment for Blight:

Emergency treatment:
1. Remove and destroy infected plant parts immediately
2. Apply copper fungicide or chlorothalonil
3. Avoid working with plants when wet
4. Use drip irrigation instead of overhead watering
5. Rotate crops annually
6. Choose resistant varieties when replanting""",
            
            "coffee leaf": """☕ Treatment for Coffee Leaf Issues:

Specialized care:
1. Inspect for specific pests or fungi
2. Apply appropriate fungicide for coffee plants
3. Ensure proper shade and moisture levels
4. Prune affected branches
5. Use organic neem oil treatment
6. Maintain soil pH between 6.0-6.5""",
            
            "apple rust": """🍎 Treatment for Apple Rust:

Comprehensive treatment:
1. Remove infected leaves and fruit immediately
2. Apply fungicide containing myclobutanil
3. Remove nearby juniper plants (alternative host)
4. Improve air circulation through pruning
5. Apply dormant spray in late winter
6. Choose rust-resistant apple varieties for future planting""",

            "root rot": """🌱 Treatment for Root Rot:

Emergency root care:
1. Remove plant from soil immediately
2. Trim away black, mushy roots
3. Treat remaining healthy roots with fungicide
4. Repot in fresh, well-draining soil
5. Reduce watering frequency significantly
6. Ensure proper drainage in containers""",

            "rust": """🦠 Treatment for Plant Rust:

Rust management:
1. Remove affected leaves immediately
2. Apply copper-based fungicide
3. Improve air circulation around plants
4. Water at soil level, not on leaves
5. Remove plant debris regularly
6. Consider resistant plant varieties"""
        }
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        if self.camera and self.camera.status() == QCamera.ActiveStatus:
            self.camera.stop()
        event.accept()

# ----------------------------
# Main Function
# ----------------------------
def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = PlantDiseaseApp()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()