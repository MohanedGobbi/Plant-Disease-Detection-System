# Plant-Disease-Detection-System
# Plant Disease Detection System

A comprehensive plant disease detection application that combines deep learning models with an intuitive PyQt5 interface. The system supports multiple input methods including image upload, camera capture, text descriptions, and symptom similarity matching.

## Features

### Multi-Modal Disease Detection
- **Image-based Detection**: Upload images or capture photos using your camera
- **Text-based Detection**: Describe symptoms in natural language
- **Symptom Similarity Matching**: Find diseases with similar symptoms using TF-IDF and SBERT

### Advanced Image Processing
- Real-time image filters for better visualization
- Support for multiple image formats (PNG, JPG, JPEG, BMP)
- Image preprocessing optimized for plant disease detection

### Intelligent Analysis
- **ResNet18 + XGBoost**: Image-based classification
- **BiLSTM**: Natural language processing for symptom descriptions
- **Dual Similarity Matching**: TF-IDF and Sentence Transformers for symptom comparison

### User-Friendly Interface
- Modern PyQt5 GUI with tabbed interface
- Real-time confidence scoring with visual indicators
- Comprehensive treatment recommendations
- Camera integration for live photo capture

## Screenshots

```
[Main Interface]
┌─────────────────────────────────────────────────────────┐
│  Upload Image  │  Take Photo  │  Text Description  │  ... │
├─────────────────┴─────────────┴───────────────────┴─────┤
│  Image Filters                    │  Image Display      │
│  ├─ Grayscale                     │  ┌─────────────────┐ │
│  ├─ Contrast                      │  │                 │ │
│  ├─ Brightness                    │  │   Plant Image   │ │
│  └─ ...more filters               │  │                 │ │
│                                   │  └─────────────────┘ │
│  [Predict Disease]                │  Prediction Results  │
│                                   │  Treatment Info      │
└───────────────────────────────────┴─────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- PyQt5
- PyTorch
- CUDA (optional, for GPU acceleration)

### Clone Repository
```bash
git clone https://github.com/MohanedGobbi/plant-disease-detection.git
cd plant-disease-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Required Models
The application requires several pre-trained models:

1. **XGBoost Model**: `models/xgb_plant_disease.pkl`
2. **ResNet18 Feature Extractor**: Download separately from [Google Drive](https://drive.google.com/file/d/1GaTmNe2cyULIJu9gH809pPfiYOwgtome/view?usp=drive_link) and place it as `models/resnet18_feature_extractor_full.pth`
3. **BiLSTM Model**: `models/bilstm_model.pth`
4. **Vocabulary and Embeddings**: 
   - `models/vocab.pkl`
   - `models/label_encoder.pkl`
   - `models/embedding_matrix.npy`

#### Manual Download Steps for ResNet18 Model:
1. Download the ResNet18 model from: [https://drive.google.com/file/d/1GaTmNe2cyULIJu9gH809pPfiYOwgtome/view?usp=drive_link](https://drive.google.com/file/d/1GaTmNe2cyULIJu9gH809pPfiYOwgtome/view?usp=drive_link)
2. Create a `models/` directory in your project root if it doesn't exist
3. Place the downloaded file as `models/resnet18_feature_extractor_full.pth`

### Required Data Files
- `data/plantwild_prompts.json` - Disease symptom descriptions for similarity matching

## Usage

### Running the Application
```bash
python app.py
```

### Input Methods

#### 1. Image Upload
- Click "Upload Image" tab
- Select an image file
- Apply optional filters for better visualization
- Click "Predict Disease"

#### 2. Camera Capture
- Click "Take Photo" tab
- Select your camera from the dropdown
- Click "Start Camera"
- Click "Capture Photo"
- Click "Predict Disease"

#### 3. Text Description
- Click "Text Description" tab
- Describe your plant's symptoms in natural language
- Example: "My tomato plant has yellow leaves with black spots"
- Click "Predict Disease"

#### 4. Symptom Matching
- Click "Symptom Matching" tab
- Choose matching method (TF-IDF, SBERT, or Both)
- Describe symptoms
- Click "Find Similar Diseases"

## Model Architecture

### Image Classification Pipeline
```
Input Image → ResNet18 Feature Extractor → XGBoost Classifier → Disease Prediction
```

### Text Classification Pipeline
```
Text Input → Preprocessing → BiLSTM → Disease Classification
```

### Similarity Matching Pipeline
```
Query Text → TF-IDF/SBERT → Cosine Similarity → Top-K Diseases
```

## Supported Diseases

The system can detect various plant diseases including:
- Powdery Mildew
- Leaf Spot
- Blight
- Rust
- Root Rot
- Coffee Leaf Diseases
- Apple Rust
- And many more...

## Project Structure

```
plant-disease-detection/
│
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
│
├── models/                         # Pre-trained models directory
│   ├── xgb_plant_disease.pkl
│   ├── resnet18_feature_extractor_full.pth  # Download separately!
│   ├── bilstm_model.pth
│   ├── vocab.pkl
│   ├── label_encoder.pkl
│   └── embedding_matrix.npy
│
├── data/                           # Data files
│   └── plantwild_prompts.json
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── similarity.py               # Symptom similarity matching
│   ├── preprocess.py               # Image/text preprocessing
│   └── filters.py                  # Image filters
│
├── docs/                           # Documentation
│   ├── user_guide.md
│   └── api_reference.md
│
└── screenshots/                    # Application screenshots
    ├── main_interface.png
    ├── image_detection.png
    ├── text_detection.png
    └── symptom_matching.png
```

## Technical Details

### Dependencies
- **PyQt5**: GUI framework
- **PyTorch**: Deep learning framework
- **Torchvision**: Computer vision utilities
- **XGBoost**: Gradient boosting classifier
- **Scikit-learn**: Machine learning utilities
- **NLTK**: Natural language processing
- **Sentence-Transformers**: Semantic similarity
- **PIL**: Image processing
- **NumPy**: Numerical computing
- **Joblib**: Model serialization

### Image Preprocessing
- Resize to 224x224 pixels
- Normalize using ImageNet statistics
- Support for RGB, grayscale, and RGBA formats

### Text Preprocessing
- Tokenization and lemmatization
- Stop word removal
- Vocabulary mapping for neural networks

### Performance
- **Image Detection**: ~2-3 seconds per image
- **Text Detection**: ~1-2 seconds per query
- **Symptom Matching**: ~1 second per query
- **Memory Usage**: ~500MB-1GB depending on models

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/MohanedGobbi/plant-disease-detection.git
cd plant-disease-detection

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 .
black .
```

## Troubleshooting

### Common Issues

#### Model Loading Errors
- Ensure all model files are in the `models/` directory
- **Important**: Make sure you've downloaded the ResNet18 model from the provided Google Drive link
- Check file permissions and paths
- Verify Python version compatibility

#### Camera Not Working
- Check camera permissions
- Ensure PyQt5 multimedia components are installed
- Try different camera indices

#### Low Prediction Confidence
- Ensure good lighting and image quality
- Try different image angles
- Use image filters to enhance visibility

#### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage if GPU issues persist
export CUDA_VISIBLE_DEVICES=""
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Scikit-learn for machine learning utilities
- NLTK for natural language processing tools
- Sentence-Transformers for semantic similarity
- The plant pathology community for disease datasets

## Contact

- GitHub: [@MohanedGobbi](https://github.com/MohanedGobbi)
- Email: gobbimouhaned@gmail.com
- Project Link: [https://github.com/MohanedGobbi/plant-disease-detection](https://github.com/MohanedGobbi/Plant-Disease-Detection-System)

## Changelog

### v1.0.0 (2024-XX-XX)
- Initial release
- Multi-modal disease detection
- PyQt5 GUI interface
- Image filtering capabilities
- Symptom similarity matching
- Comprehensive treatment recommendations
