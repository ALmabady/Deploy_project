from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
import io
import base64

# ------------------ Model Definition ------------------

class PrototypicalNetwork(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        self.embedding_layer = torch.nn.Linear(self.backbone.embed_dim, embedding_dim)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.ndim == 3:
            features = features[:, 0, :]
        embedding = self.embedding_layer(features)
        return F.normalize(embedding, p=2, dim=1)

# ------------------ Load Model ------------------

model = PrototypicalNetwork(embedding_dim=128)
model.load_state_dict(torch.load("model_state.pth", map_location=torch.device("cpu")))
model.eval()

# ------------------ Load Class Prototypes ------------------

class_prototypes = torch.load("class_prototypes.pth", map_location=torch.device("cpu"))

# ------------------ Preprocessing & Classes ------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

# ------------------ FastAPI App ------------------

app = FastAPI()

@app.post("/api/predict_base64")
async def predict_base64(request: Request):
    try:
        data = await request.json()
        base64_str = data.get("image")

        if not base64_str:
            return JSONResponse(status_code=400, content={"error": "No image data provided."})

        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model(image_tensor)

        distances = torch.cdist(embedding, class_prototypes)
        predicted_class_idx = torch.argmin(distances).item()
        predicted_class = class_names[predicted_class_idx]
        distance = distances[0, predicted_class_idx].item()

        if distance > threshold:
            return JSONResponse(content={"prediction": "Uncertain, distance too high"})

        return JSONResponse(content={"prediction": predicted_class, "distance": distance})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
