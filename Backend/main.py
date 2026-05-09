"""
NEOSCAN AI — FastAPI Backend
Serves EfficientNetB4 (Eye) + EfficientNetB3 (Skin) + ResNet50 (Body) + VGG16+ViT (Face)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
import os
import io
from PIL import Image

app = FastAPI(title="NeoScan AI API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
#  EYE MODULE — TensorFlow / EfficientNetB4
# ═══════════════════════════════════════════════════════════════
EYE_MODEL  = None
EYE_CONFIG = None

def load_eye_model():
    global EYE_MODEL, EYE_CONFIG
    try:
        import tensorflow as tf
        MODEL_PATH  = os.environ.get("EYE_MODEL_PATH",  "neoscan_eye_FINAL_effb4.keras")
        CONFIG_PATH = os.environ.get("EYE_CONFIG_PATH", "eye_model_config_FINAL.json")
        if os.path.exists(MODEL_PATH):
            EYE_MODEL = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Eye model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  Eye model not found at {MODEL_PATH} — running in DEMO mode")
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                EYE_CONFIG = json.load(f)
            print(f"✅ Eye config loaded")
        else:
            EYE_CONFIG = {
                "idx_to_class": {
                    "0": "cataract", "1": "normal_eye", "2": "stage_0_normal",
                    "3": "stage_1",  "4": "stage_2",   "5": "stage_3"
                }
            }
    except Exception as e:
        print(f"⚠️  Eye model load error: {e} — running in DEMO mode")

load_eye_model()

EYE_RISK = {
    "cataract"       : {"level": "HIGH",   "color": "#ef4444", "advice": "Immediate ophthalmology referral required. Cataract or leukocoria detected."},
    "normal_eye"     : {"level": "NORMAL", "color": "#22c55e", "advice": "No eye abnormalities detected. Continue routine monitoring."},
    "stage_0_normal" : {"level": "NORMAL", "color": "#22c55e", "advice": "No ROP detected. Schedule next screening as per protocol."},
    "stage_1"        : {"level": "LOW",    "color": "#f59e0b", "advice": "Mild ROP (Stage 1) detected. Weekly monitoring recommended."},
    "stage_2"        : {"level": "MEDIUM", "color": "#f97316", "advice": "Moderate ROP (Stage 2) detected. Urgent ophthalmology review needed within 72 hours."},
    "stage_3"        : {"level": "HIGH",   "color": "#ef4444", "advice": "Severe ROP (Stage 3 / AP-ROP) detected. Immediate treatment required."},
}

EYE_LABELS = {
    "cataract"       : "Cataract / Leukocoria",
    "normal_eye"     : "Normal Eye",
    "stage_0_normal" : "ROP Stage 0 — No ROP",
    "stage_1"        : "ROP Stage 1 — Mild",
    "stage_2"        : "ROP Stage 2 — Moderate",
    "stage_3"        : "ROP Stage 3 — Severe",
}

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def preprocess_eye(image_bytes, image_type="rop"):
    import tensorflow as tf
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    if image_type == "rop":
        img = cv2.addWeighted(img, 4,
                  cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    img = cv2.resize(img, (380, 380))
    img = apply_clahe(img)
    img = tf.keras.applications.efficientnet.preprocess_input(
              img.astype("float32"))
    return np.expand_dims(img, axis=0)

# ═══════════════════════════════════════════════════════════════
#  SKIN MODULE — PyTorch / EfficientNetB3
# ═══════════════════════════════════════════════════════════════
SKIN_MODEL  = None
SKIN_DEVICE = None

SKIN_CLASS_NAMES = ['anemia', 'cyanosis', 'jaundice', 'normal']
SKIN_IDX_TO_CLASS = {0: 'anemia', 1: 'cyanosis', 2: 'jaundice', 3: 'normal'}

SKIN_RISK = {
    "anemia"   : {"level": "MODERATE", "color": "#f97316", "advice": "Obtain CBC. Check haemoglobin. Assess for haemolysis. Pale skin tone suggests possible anaemia."},
    "cyanosis" : {"level": "CRITICAL", "color": "#ef4444", "advice": "URGENT: Check O₂ saturation immediately. May indicate cardiac or respiratory issue. Escalate now."},
    "jaundice" : {"level": "MODERATE", "color": "#f59e0b", "advice": "Check bilirubin levels. Consider phototherapy if bilirubin >15 mg/dL. Monitor closely."},
    "normal"   : {"level": "NORMAL",   "color": "#22c55e", "advice": "Skin tone appears normal. No signs of jaundice, cyanosis or anaemia detected. Routine monitoring."},
}

SKIN_LABELS = {
    "anemia"   : "Anaemia",
    "cyanosis" : "Cyanosis",
    "jaundice" : "Jaundice",
    "normal"   : "Normal Skin",
}

# ImageNet normalisation (same as training)
SKIN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SKIN_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_skin_model():
    global SKIN_MODEL, SKIN_DEVICE
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        SKIN_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MODEL_PATH  = os.environ.get("SKIN_MODEL_PATH", "best_model.pth")

        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Skin model not found at {MODEL_PATH} — running in DEMO mode")
            return

        # Rebuild the same architecture used in training
        class NeoScanSkinModel(nn.Module):
            def __init__(self, num_classes=4, freeze_ratio=0.7):
                super().__init__()
                backbone = efficientnet_b3(weights=None)
                self.features   = backbone.features
                self.avgpool    = backbone.avgpool
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.4, inplace=True),
                    nn.Linear(backbone.classifier[1].in_features, num_classes)
                )
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.classifier(x)

        model = NeoScanSkinModel(num_classes=4).to(SKIN_DEVICE)
        ckpt  = torch.load(MODEL_PATH, map_location=SKIN_DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        SKIN_MODEL = model
        print(f"✅ Skin model loaded from {MODEL_PATH}  (epoch {ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc',0):.4f})")
    except Exception as e:
        print(f"⚠️  Skin model load error: {e} — running in DEMO mode")

load_skin_model()

def white_balance(img):
    """Gray World white balance — corrects warm/cool lighting bias."""
    img_float = img.astype(np.float32)
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (avg_gray / avg_b), 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (avg_gray / avg_g), 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return img_float.astype(np.uint8)

def apply_skin_clahe(img):
    """Apply CLAHE on L channel in LAB space — same as training pipeline."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def preprocess_skin(image_bytes):
    """
    Robust skin preprocessing for real-world images:
    1. White balance  — fixes warm/cool/phototherapy lighting
    2. CLAHE          — matches training pipeline exactly
    3. Resize 224x224 — EfficientNetB3 input size
    4. ImageNet norm  — same as training
    """
    import torch
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # Step 1 — White balance to fix lighting color cast
    img = white_balance(img)

    # Step 2 — CLAHE for contrast normalisation (matches training)
    img = apply_skin_clahe(img)

    # Step 3 — Convert to RGB and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Step 4 — ImageNet normalisation
    img = img.astype(np.float32) / 255.0
    img = (img - SKIN_MEAN) / SKIN_STD
    img = np.transpose(img, (2, 0, 1))            # HWC → CHW
    tensor = torch.from_numpy(img).unsqueeze(0).to(SKIN_DEVICE)
    return tensor

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "NeoScan AI API running", "version": "2.0.0"}

@app.get("/health")
def health():
    return {
        "status"            : "ok",
        "eye_model_loaded"  : EYE_MODEL  is not None,
        "skin_model_loaded" : SKIN_MODEL is not None,
        "body_model_loaded" : BODY_MODEL is not None,
        "face_model_loaded" : FACE_VGG is not None or FACE_VIT is not None,
        "eye_mode"          : "live" if EYE_MODEL  is not None else "demo",
        "skin_mode"         : "live" if SKIN_MODEL is not None else "demo",
        "body_mode"         : "live" if BODY_MODEL is not None else "demo",
        "face_mode"         : "live" if FACE_VGG is not None or FACE_VIT is not None else "demo",
    }

# ── EYE PREDICTION ────────────────────────────────────────────
@app.post("/predict/eye")
async def predict_eye(
    file: UploadFile = File(...),
    image_type: str = "rop"
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()

    # DEMO fallback
    if EYE_MODEL is None:
        import random
        classes = list(EYE_RISK.keys())
        pred    = random.choice(classes)
        probs   = np.random.dirichlet(np.ones(6)).tolist()
        risk    = EYE_RISK[pred]
        return {
            "prediction"   : pred,
            "label"        : EYE_LABELS[pred],
            "confidence"   : round(max(probs) * 100, 2),
            "risk_level"   : risk["level"],
            "risk_color"   : risk["color"],
            "advice"       : risk["advice"],
            "probabilities": {EYE_LABELS[c]: round(p * 100, 2) for c, p in zip(classes, probs)},
            "mode"         : "demo"
        }

    # LIVE prediction
    try:
        img        = preprocess_eye(image_bytes, image_type)
        probs      = EYE_MODEL.predict(img, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        idx_map    = EYE_CONFIG.get("idx_to_class", {})
        pred_class = idx_map.get(str(pred_idx), "unknown")
        risk       = EYE_RISK.get(pred_class, EYE_RISK["normal_eye"])
        probs_dict = {
            EYE_LABELS.get(idx_map.get(str(i), ""), str(i)): round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }
        return {
            "prediction"   : pred_class,
            "label"        : EYE_LABELS.get(pred_class, pred_class),
            "confidence"   : round(float(probs[pred_idx]) * 100, 2),
            "risk_level"   : risk["level"],
            "risk_color"   : risk["color"],
            "advice"       : risk["advice"],
            "probabilities": probs_dict,
            "mode"         : "live"
        }
    except Exception as e:
        raise HTTPException(500, f"Eye prediction error: {str(e)}")

# ── SKIN PREDICTION ───────────────────────────────────────────
@app.post("/predict/skin")
async def predict_skin(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()

    # DEMO fallback
    if SKIN_MODEL is None:
        import random
        classes = SKIN_CLASS_NAMES
        pred    = random.choice(classes)
        probs   = np.random.dirichlet(np.ones(4)).tolist()
        risk    = SKIN_RISK[pred]
        return {
            "prediction"   : pred,
            "label"        : SKIN_LABELS[pred],
            "confidence"   : round(max(probs) * 100, 2),
            "risk_level"   : risk["level"],
            "risk_color"   : risk["color"],
            "advice"       : risk["advice"],
            "probabilities": {SKIN_LABELS[c]: round(p * 100, 2) for c, p in zip(classes, probs)},
            "mode"         : "demo"
        }

    # LIVE prediction
    try:
        import torch
        tensor     = preprocess_skin(image_bytes)
        with torch.no_grad():
            logits = SKIN_MODEL(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_class = SKIN_IDX_TO_CLASS[pred_idx]
        risk       = SKIN_RISK[pred_class]

        probs_dict = {
            SKIN_LABELS[SKIN_IDX_TO_CLASS[i]]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }
        return {
            "prediction"   : pred_class,
            "label"        : SKIN_LABELS[pred_class],
            "confidence"   : round(float(probs[pred_idx]) * 100, 2),
            "risk_level"   : risk["level"],
            "risk_color"   : risk["color"],
            "advice"       : risk["advice"],
            "probabilities": probs_dict,
            "mode"         : "live"
        }
    except Exception as e:
        raise HTTPException(500, f"Skin prediction error: {str(e)}")


# ═══════════════════════════════════════════════════════════════
#  BODY MODULE — PyTorch / ResNet50
# ═══════════════════════════════════════════════════════════════
BODY_MODEL  = None
BODY_DEVICE = None

BODY_RISK = {
    "normal"   : {"level": "NORMAL",   "color": "#22c55e",
                  "advice": "Body posture appears normal. Limb symmetry detected. Routine monitoring."},
    "abnormal" : {"level": "HIGH",     "color": "#ef4444",
                  "advice": "Abnormal body posture detected. Possible limb asymmetry, hydrops or restricted movement. Immediate pediatric review required."},
}

BODY_LABELS = {
    "normal"  : "Normal Posture",
    "abnormal": "Abnormal Posture",
}

BODY_MEAN = [0.485, 0.456, 0.406]
BODY_STD  = [0.229, 0.224, 0.225]

def load_body_model():
    global BODY_MODEL, BODY_DEVICE
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        BODY_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_PATH  = os.environ.get("BODY_MODEL_PATH", "body_model.pth")

        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Body model not found at {MODEL_PATH} — running in DEMO mode")
            return

        class NeoScanNet(nn.Module):
            def __init__(self):
                super().__init__()
                base = models.resnet50(weights=None)
                self.backbone = nn.Sequential(*list(base.children())[:-1])
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 128), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2)
                )
            def forward(self, x):
                return self.head(self.backbone(x))

        model = NeoScanNet().to(BODY_DEVICE)
        ckpt  = torch.load(MODEL_PATH, map_location=BODY_DEVICE)

        # Handle both raw state dict and wrapped checkpoint
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
        BODY_MODEL = model
        print(f"✅ Body model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Body model load error: {e} — running in DEMO mode")

load_body_model()

def preprocess_body(image_bytes):
    """Resize to 224x224, ImageNet normalise — same as val_tf in training."""
    import torch
    import torchvision.transforms as transforms
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    tf  = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(BODY_MEAN, BODY_STD),
    ])
    return tf(pil).unsqueeze(0).to(BODY_DEVICE)

# ── BODY PREDICTION ───────────────────────────────────────────
@app.post("/predict/body")
async def predict_body(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()

    # DEMO fallback
    if BODY_MODEL is None:
        import random
        pred   = random.choice(["normal", "abnormal"])
        probs  = np.random.dirichlet(np.ones(2)).tolist()
        risk   = BODY_RISK[pred]
        return {
            "prediction"   : pred,
            "label"        : BODY_LABELS[pred],
            "confidence"   : round(max(probs) * 100, 2),
            "risk_level"   : risk["level"],
            "risk_color"   : risk["color"],
            "advice"       : risk["advice"],
            "probabilities": {
                "Normal Posture"  : round(probs[0] * 100, 2),
                "Abnormal Posture": round(probs[1] * 100, 2),
            },
            "mode": "demo"
        }

    # LIVE prediction
    try:
        import torch
        tensor = preprocess_body(image_bytes)
        with torch.no_grad():
            logits = BODY_MODEL(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_class = "normal" if pred_idx == 0 else "abnormal"
        risk       = BODY_RISK[pred_class]

        return {
            "prediction"   : pred_class,
            "label"        : BODY_LABELS[pred_class],
            "confidence"   : round(float(probs[pred_idx]) * 100, 2),
            "risk_level"   : risk["level"],
            "risk_color"   : risk["color"],
            "advice"       : risk["advice"],
            "probabilities": {
                "Normal Posture"  : round(float(probs[0]) * 100, 2),
                "Abnormal Posture": round(float(probs[1]) * 100, 2),
            },
            "mode": "live"
        }
    except Exception as e:
        raise HTTPException(500, f"Body prediction error: {str(e)}")


# ═══════════════════════════════════════════════════════════════
#  FACE MODULE — PyTorch / VGG16 + ViT Ensemble
# ═══════════════════════════════════════════════════════════════
FACE_VGG    = None
FACE_VIT    = None
FACE_DEVICE = None

FACE_CLASS_NAMES = ['Cleft', 'DownSyndrome', 'Healthy']
FACE_IDX2LABEL   = {0: 'Cleft', 1: 'DownSyndrome', 2: 'Healthy'}
FACE_MEAN = [0.485, 0.456, 0.406]
FACE_STD  = [0.229, 0.224, 0.225]
IMG_SIZE_FACE = 224

FACE_RISK = {
    'Cleft'       : {'level': 'HIGH',   'color': '#ef4444',
                     'advice': 'Cleft lip/palate detected. Refer to craniofacial surgery team immediately. Pre-surgical evaluation recommended.'},
    'DownSyndrome': {'level': 'HIGH',   'color': '#f97316',
                     'advice': 'Down Syndrome facial features detected. Refer to genetics team. Cardiac screening and developmental assessment required.'},
    'Healthy'     : {'level': 'NORMAL', 'color': '#22c55e',
                     'advice': 'No facial abnormalities detected. Normal neonatal facial features. Routine monitoring recommended.'},
}

FACE_LABELS = {
    'Cleft'       : 'Cleft Lip / Palate',
    'DownSyndrome': 'Down Syndrome',
    'Healthy'     : 'Healthy — Normal',
}

def load_face_models():
    global FACE_VGG, FACE_VIT, FACE_DEVICE
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        FACE_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── VGG16 ─────────────────────────────────────────────
        class VGG16Classifier(nn.Module):
            def __init__(self, num_classes=3, dropout=0.6):
                super().__init__()
                vgg = models.vgg16(weights=None)
                self.features   = vgg.features
                self.avgpool    = vgg.avgpool
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * 7 * 7, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(0.65),
                    nn.Linear(1024, 512),          nn.BatchNorm1d(512),  nn.ReLU(inplace=True), nn.Dropout(0.5),
                    nn.Linear(512, 128),           nn.ReLU(inplace=True), nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
            def forward(self, x):
                x = self.features(x); x = self.avgpool(x); return self.classifier(x)

        vgg_path = os.environ.get('FACE_VGG_PATH', 'VGG16_best.pth')
        if os.path.exists(vgg_path):
            vgg = VGG16Classifier(num_classes=3).to(FACE_DEVICE)
            ckpt = torch.load(vgg_path, map_location=FACE_DEVICE)
            vgg.load_state_dict(ckpt.get('model_state_dict', ckpt))
            vgg.eval()
            FACE_VGG = vgg
            print(f'✅ Face VGG16 loaded from {vgg_path}')
        else:
            print(f'⚠️  Face VGG16 not found at {vgg_path}')

        # ── ViT ───────────────────────────────────────────────
        try:
            import timm

            class ViTFaceClassifier(nn.Module):
                def __init__(self, num_classes=3, dropout=0.3):
                    super().__init__()
                    self.vit  = timm.create_model('vit_small_patch16_224', pretrained=False,
                                                   num_classes=0, drop_rate=dropout)
                    embed_dim = self.vit.embed_dim
                    self.head = nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(dropout),
                        nn.Linear(256, num_classes)
                    )
                def forward(self, x):
                    return self.head(self.vit(x))

            vit_path = os.environ.get('FACE_VIT_PATH', 'ViT_v2_best.pth')
            if os.path.exists(vit_path):
                vit = ViTFaceClassifier(num_classes=3).to(FACE_DEVICE)
                ckpt = torch.load(vit_path, map_location=FACE_DEVICE)
                vit.load_state_dict(ckpt.get('model_state_dict', ckpt))
                vit.eval()
                FACE_VIT = vit
                print(f'✅ Face ViT loaded from {vit_path}')
            else:
                print(f'⚠️  Face ViT not found at {vit_path}')
        except ImportError:
            print('⚠️  timm not installed — ViT not loaded. Run: pip install timm')

    except Exception as e:
        print(f'⚠️  Face model load error: {e} — running in DEMO mode')

load_face_models()

def preprocess_face(image_bytes):
    """Preprocess face image — 224x224 ImageNet normalised."""
    import torch
    import torchvision.transforms as T
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    tf  = T.Compose([
        T.Resize((IMG_SIZE_FACE, IMG_SIZE_FACE)),
        T.ToTensor(),
        T.Normalize(FACE_MEAN, FACE_STD),
    ])
    return tf(pil).unsqueeze(0).to(FACE_DEVICE)

@app.post('/predict/face')
async def predict_face(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, 'File must be an image')
    image_bytes = await file.read()

    # DEMO fallback
    if FACE_VGG is None and FACE_VIT is None:
        import random
        pred  = random.choice(FACE_CLASS_NAMES)
        probs = np.random.dirichlet(np.ones(3)).tolist()
        risk  = FACE_RISK[pred]
        return {
            'prediction'   : pred,
            'label'        : FACE_LABELS[pred],
            'confidence'   : round(max(probs) * 100, 2),
            'risk_level'   : risk['level'],
            'risk_color'   : risk['color'],
            'advice'       : risk['advice'],
            'probabilities': {FACE_LABELS[c]: round(p * 100, 2) for c, p in zip(FACE_CLASS_NAMES, probs)},
            'mode'         : 'demo',
            'model_used'   : 'demo'
        }

    # LIVE prediction — ensemble VGG16 + ViT
    try:
        import torch
        tensor = preprocess_face(image_bytes)
        probs_combined = None

        with torch.no_grad():
            if FACE_VGG is not None:
                vgg_probs = torch.softmax(FACE_VGG(tensor), dim=1).cpu().numpy()[0]
                probs_combined = vgg_probs

            if FACE_VIT is not None:
                vit_probs = torch.softmax(FACE_VIT(tensor), dim=1).cpu().numpy()[0]
                if probs_combined is not None:
                    probs_combined = 0.5 * probs_combined + 0.5 * vit_probs
                else:
                    probs_combined = vit_probs

        pred_idx   = int(np.argmax(probs_combined))
        pred_class = FACE_IDX2LABEL[pred_idx]
        risk       = FACE_RISK[pred_class]
        mode       = 'ensemble' if (FACE_VGG and FACE_VIT) else ('vgg16' if FACE_VGG else 'vit')

        return {
            'prediction'   : pred_class,
            'label'        : FACE_LABELS[pred_class],
            'confidence'   : round(float(probs_combined[pred_idx]) * 100, 2),
            'risk_level'   : risk['level'],
            'risk_color'   : risk['color'],
            'advice'       : risk['advice'],
            'probabilities': {FACE_LABELS[FACE_IDX2LABEL[i]]: round(float(p) * 100, 2)
                              for i, p in enumerate(probs_combined)},
            'mode'         : 'live',
            'model_used'   : mode
        }
    except Exception as e:
        raise HTTPException(500, f'Face prediction error: {str(e)}')
