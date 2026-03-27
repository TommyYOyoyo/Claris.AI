import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Page setup
st.set_page_config(page_title="Claris.AI", page_icon="🔬")
st.title("Claris.AI: diagnostic de cancer")
st.write("Téléversez un scan de 96x96 afin de détecter la presence de tissu métastatique.")

# Load model once
@st.cache_resource
def load_model():
    # Recreate the ResNet-50 architecture with 2 output classes
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load the weights. Use CPU so it runs on any laptop.
    state_dict = torch.load("/Users/tommy/Tommy/Programming/Projects/Claris/src/saves/claris_resnet-50.pth", map_location=torch.device('cpu'))
    
    # Removes "module." prefix from keys so it works locally.
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict) # Load the weights
    model.eval() # Set to evaluation mode
    return model # Return the model

# Load the model
model = load_model()

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
    # Add a batch dimension (C, H, W) -> (1, C, H, W)
    return val_tfms(image).unsqueeze(0) 

# GUI
uploaded_file = st.file_uploader("Choisissez une image...", type=["tif", "png", "jpg", "jpeg"])

# If uploaded file sucessfully uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Check image size and warn if it's not 96x96
    if image.size != (96, 96):
        st.warning(f"Note: Image est de format {image.size[0]}x{image.size[1]}. Ce modèle a été entraîné sur des images 96x96, donc les résultats peuvent varier.")
    
    # Show the image on screen
    st.image(image, caption='Scan téléversé', width=300)
    
    st.write("### Analysant l'image...")
    
    # Prepare the image and predict
    tensor = transform_image(image)
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