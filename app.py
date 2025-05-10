import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import base64
import io
from flask import Flask, request, jsonify

# ----- Model Definition -----
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
        self.embedding_layer = nn.Linear(self.backbone.embed_dim, embedding_dim)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.ndim == 3:
            features = features[:, 0, :]
        embedding = self.embedding_layer(features)
        return F.normalize(embedding, p=2, dim=1)

# ----- Load Model & Prototypes -----
model = PrototypicalNetwork(embedding_dim=128)
model.load_state_dict(torch.load("model_state.pth", map_location="cpu"))
model.eval()

class_prototypes = torch.load("class_prototypes.pth", map_location="cpu")

# ----- Class Labels -----
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

# ----- Image Preprocessing -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return transform(image).unsqueeze(0)

# ----- Inference Logic -----
def predict_image(image_tensor):
    with torch.no_grad():
        embedding = model(image_tensor)
        distances = {
            cls: torch.norm(embedding - proto.unsqueeze(0)).item()
            for cls, proto in class_prototypes.items()
        }
        pred_class = min(distances, key=distances.get)
        if distances[pred_class] > threshold:
            return "unknown"
        return class_names[pred_class]

# ----- Flask App -----
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Fas7ni Tourism Detector API"}), 200

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_tensor = preprocess_base64_image(data["image"])
        prediction = predict_image(image_tensor)
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
