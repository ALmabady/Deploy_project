from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import io

# ---- Minimal CNN Encoder ----
class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, embedding_dim)

    def forward(self, x):
        x = self.conv(x).squeeze()
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# ---- Load model + prototypes ----
model = SimpleCNN(embedding_dim=128)
model.load_state_dict(torch.load("model_state.pth", map_location=torch.device("cpu")))
model.eval()

class_prototypes = torch.load("class_prototypes.pth", map_location=torch.device("cpu"))

# ---- Class labels & threshold ----
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

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ---- FastAPI setup ----
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img_tensor)

        distances = {
            cls: torch.norm(embedding - proto.unsqueeze(0)).item()
            for cls, proto in class_prototypes.items()
        }
        pred_class = min(distances, key=distances.get)
        min_distance = distances[pred_class]

        if min_distance > threshold:
            return JSONResponse(content={"prediction": "unknown"})
        return JSONResponse(content={"prediction": class_names[pred_class]})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
