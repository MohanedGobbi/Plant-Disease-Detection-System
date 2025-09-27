import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess image for model prediction
    Same preprocessing as used during training
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

def extract_features(model, image_tensor, device):
    """
    Extract features from image using the model
    """
    with torch.no_grad():
        feat = model(image_tensor.to(device))
        feat = feat.view(feat.size(0), -1).cpu().numpy()
    return feat

def pil_to_qimage(pil_image):
    """
    Convert PIL image to QImage for display in PyQt
    """
    # Convert PIL image to numpy array
    if pil_image.mode == "RGB":
        r, g, b = pil_image.split()
        pil_image = Image.merge("RGB", (b, g, r))
    elif pil_image.mode == "RGBA":
        r, g, b, a = pil_image.split()
        pil_image = Image.merge("RGBA", (b, g, r, a))
    elif pil_image.mode == "L":
        pil_image = pil_image.convert("RGB")
        r, g, b = pil_image.split()
        pil_image = Image.merge("RGB", (b, g, r))
    
    # Convert to QImage
    data = pil_image.tobytes("raw", "RGB")
    qimage = ImageQt.ImageQt(pil_image)
    
    return qimage

    
    
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"[^a-z\s]", "", text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return tokens