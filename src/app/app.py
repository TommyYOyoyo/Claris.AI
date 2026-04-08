import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Page setup
st.set_page_config(page_title="Claris.AI", page_icon="🔬")
st.title("Claris.AI")
st.write("Téléversez un scan de 96x96 afin de détecter la presence de tissu métastatique.")

MODEL_OPTIONS = {
    "ResNet-50 (98.03% / rapide) - Par défaut": {
        "id": "resnet50",
        "weights": "claris_default_resnet-50.pth"
    },
    "DenseNet-121 (97.99% / moyen)": {
        "id": "densenet121",
        "weights": "claris_densenet-121.pth"
    },
    "EfficientNetV2-L (98.13% / lent)": {
        "id": "efficientnet_v2_l",
        "weights": "claris_efficientnetV2.pth"
    },
    "VGG16-BN (97.39% / rapide)": {
        "id": "vgg16_bn",
        "weights": "claris_vgg16_bn.pth"
    }
}

# Model selector
selected_model_label = st.selectbox("Choisissez un modèle :", tuple(MODEL_OPTIONS.keys()))

# Model builder for each architecture
def build_model(model_id):
    if model_id == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if model_id == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model

    if model_id == "efficientnet_v2_l":
        model = models.efficientnet_v2_l(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    if model_id == "vgg16_bn":
        model = models.vgg16_bn(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    raise ValueError(f"Unsupported model id: {model_id}")

# Get last convolutional layer for each architecture
def get_last_conv_layer(model):
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layer found for Grad-CAM target layer.")
    return conv_layers[-1]

# Load model once
@st.cache_resource
def load_model(model_id, weights_filename):
    model = build_model(model_id)

    # Load the saved weights
    state_dict = torch.load(Path(__file__).parent / f"../saves/{weights_filename}", map_location=torch.device('cpu'))
    
    # Remove "module." prefix from keys so it works locally
    clean_state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }
    
    model.load_state_dict(clean_state_dict) # Load the weights
    model.eval() # Set to evaluation mode

    # Grad-CAM can use the deepest convolutional layer across architectures
    target_layer = get_last_conv_layer(model)
    return model, [target_layer]

# Apply same transformations used in validation loop
def transform_image(image):
    val_tfms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Add a batch dimension (Channel, Height, Width) -> (1, Channel, Height, Width)
    return val_tfms(image).unsqueeze(0)

# Image resizing to prevent errors
def resize_image(image):
    return image.resize((96, 96), Image.Resampling.BILINEAR)


# Load the model
selected_model_config = MODEL_OPTIONS[selected_model_label]
model, target_layers = load_model(selected_model_config["id"], selected_model_config["weights"])

# GUI
uploaded_file = st.file_uploader("Choisissez une image...", type=["tif", "png", "jpg", "jpeg"])

# If uploaded file sucessfully uploaded
if uploaded_file is not None:
    
    # Display the uploaded image along with GradCAM heatmap
    col1, col2 = st.columns(2)
    
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Check image size and warn if it's not 96x96
    if image.size != (96, 96):
        st.warning(f"Note: Image est de format {image.size[0]}x{image.size[1]}. Ce modèle a été entraîné sur des images 96x96, donc les résultats peuvent varier.")
    
    # Show the image on screen
    
    with col1:
        st.image(image, caption='Scan téléversé', width=300)
    
    # To be erased after
    placeholder = st.empty()
    placeholder.write("### Analysant l'image...")
    
    # Prepare the image and predict
    resized_image = resize_image(image)
    tensor = transform_image(resized_image)
    with torch.no_grad():
        outputs = model(tensor)
        # Get percentages using Softmax
        probabilities = F.softmax(outputs, dim=1)[0]
        # Get the predicted class (0 or 1)
        prediction = torch.argmax(outputs, dim=1).item()
        
    # Display results
    classes = ["Benigne (ø Cancer)", "Maligne (Cancer Détecté)"]
    confidence = probabilities[prediction].item() * 100
    
    if prediction == 1:
        st.error(f"## Résultat: {classes[prediction]}")
    else:
        st.success(f"## Résultat: {classes[prediction]}")
        
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    placeholder.empty() # Clear progress message
    
    print(prediction)

    # Grad-CAM visualization
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(prediction)] # Target is the predicted class (0 = benign, 1 = malignant)
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0] 

    # Convert image to float RGB in [0, 1] before overlaying CAM
    rgb_image = np.array(resized_image).astype(np.float32) / 255.0
    cam_overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    
    # Display the heatmap alongside the original image
    with col2:
        st.image(cam_overlay, caption="Zones influençant la prédiction (Grad-CAM)", width=300)