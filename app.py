from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
import io
import numpy as np
import base64

# ----- Model Architecture -----
class PrototypicalNetwork(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super(PrototypicalNetwork, self).__init__()
        # Load the backbone model (DeiT small)
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, 'blocks'):
            # Unfreeze the last two blocks
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
        # Embedding layer
        self.embedding_layer = torch.nn.Linear(self.backbone.embed_dim, embedding_dim)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.ndim == 3:
            features = features[:, 0, :]
        embedding = self.embedding_layer(features)
        return F.normalize(embedding, p=2, dim=1)

# ----- Load Model and Prototypes -----
model = PrototypicalNetwork(embedding_dim=128)
state_dict = torch.load("model_state.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
# Load class prototypes
class_prototypes = torch.load("class_prototypes.pth", map_location=torch.device("cpu"))
# Set the model to evaluation mode
model.eval()

# ----- Image Preprocessing Transform -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- Class Names and Threshold -----
class_names = [
    'Abu Simbel Temple', 'Bibliotheca Alexandrina', 'Nefertari Temple', 
    'Saint Catherine Monastery', 'Citadel of Saladin', 'Monastery of St. Simeon', 
    'AlAzhar Mosque', 'Fortress of Shali in Siwa', 'Greek Orthodox Cemetery in Alexandria', 
    'Hanging Church', 'khan el khalili', 'Luxor Temple', 'Baron Empain Palace', 
    'New Alamein City', 'Philae Temple', 'Pyramid of Djoser', 'Salt lake at Siwa', 
    'Wadi Al-Hitan', 'White Desert', 'Cairo Opera House', 'Tahrir Square', 
    'Cairo tower', 'Citadel of Qaitbay', 'Egyptian Museum in Tahrir', 
    'Great Pyramids of Giza', 'Hatshepsut temple', 'Meidum pyramid', 
    'Royal Montaza Palace'
]
threshold = 0.35

# Initialize FastAPI app
app = FastAPI(title="Egyptian Landmarks Classifier API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def process_image(image):
    """Process image through the model and return prediction"""
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Get image embeddings from the model
    with torch.no_grad():
        embedding = model(image_tensor)
    
    # Calculate the distances to each class prototype
    distances = torch.cdist(embedding, class_prototypes)
    
    # Get the predicted class (minimum distance)
    predicted_class_idx = torch.argmin(distances).item()
    
    # Get the class name
    predicted_class = class_names[predicted_class_idx]
    
    # Check if the predicted class is within the threshold
    distance = distances[0, predicted_class_idx].item()
    if distance > threshold:
        return {"prediction": "Uncertain, distance too high"}
    
    return {"prediction": predicted_class, "distance": float(distance)}

@app.get("/")
def root():
    """Root endpoint to check if API is running"""
    return {"message": "Egyptian Landmarks Classifier API is running. Use /predict/ or /api/predict_base64 endpoints."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint for prediction using file upload"""
    try:
        # Read the file and convert to image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Process the image
        result = process_image(image)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/predict_base64")
async def predict_base64(data: dict = Body(...)):
    """Endpoint for prediction using base64 encoded image"""
    try:
        # Extract the base64 string from the request
        base64_string = data.get("image", "")
        if not base64_string:
            return JSONResponse(status_code=400, content={"error": "No image provided"})
        
        # Decode the base64 string to binary
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process the image
        result = process_image(image)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
