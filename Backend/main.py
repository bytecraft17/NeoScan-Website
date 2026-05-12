"""
NEOSCAN AI — FastAPI Backend
Serves EfficientNetB4 (Eye) + EfficientNetB3 (Skin) + ResNet50 (Body) + VGG16+ViT (Face)
LAZY LOADING: models load on first request — fixes Render startup timeout.
AUTO DOWNLOAD: model files downloaded from Hugging Face on first use.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
import os
from PIL import Image

# ═══════════════════════════════════════════════════════════════
#  HUGGING FACE AUTO-DOWNLOAD
# ═══════════════════════════════════════════════════════════════
HF_REPO_ID = "amine/neoscan-models"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

HF_FILES = [
    "neoscan_eye_FINAL_effb4.keras",
    "eye_model_config_FINAL.json",
    "best_model.pth",
    "body_model.pth",
    "VGG16_best.pth",
    "ViT_v2_best.pth",
]

def ensure_file(filename: str):
    """Download from Hugging Face if file not present locally."""
    if os.path.exists(filename):
        return
    try:
        from huggingface_hub import hf_hub_download
        print(f"⬇️  Downloading {filename} from Hugging Face...")
        path = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = filename,
            token     = HF_TOKEN,
            repo_type = "model",
            local_dir = "."
        )
        print(f"✅ {filename} ready ({os.path.getsize(path)/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

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
EYE_LOADED = False

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

def ensure_eye_loaded():
    global EYE_MODEL, EYE_CONFIG, EYE_LOADED
    if EYE_LOADED:
        return
    EYE_LOADED = True
    ensure_file("neoscan_eye_FINAL_effb4.keras")
    ensure_file("eye_model_config_FINAL.json")
    try:
        import tensorflow as tf
        if os.path.exists("neoscan_eye_FINAL_effb4.keras"):
            EYE_MODEL = tf.keras.models.load_model("neoscan_eye_FINAL_effb4.keras")
            print("✅ Eye model loaded")
        else:
            print("⚠️  Eye model not found — DEMO mode")
        if os.path.exists("eye_model_config_FINAL.json"):
            with open("eye_model_config_FINAL.json") as f:
                EYE_CONFIG = json.load(f)
        else:
            EYE_CONFIG = {"idx_to_class": {"0":"cataract","1":"normal_eye","2":"stage_0_normal","3":"stage_1","4":"stage_2","5":"stage_3"}}
    except Exception as e:
        print(f"⚠️  Eye load error: {e} — DEMO mode")
        EYE_CONFIG = {"idx_to_class": {"0":"cataract","1":"normal_eye","2":"stage_0_normal","3":"stage_1","4":"stage_2","5":"stage_3"}}

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def preprocess_eye(image_bytes, image_type="rop"):
    import tensorflow as tf
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    if image_type == "rop":
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    img = cv2.resize(img, (380, 380))
    img = apply_clahe(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype("float32"))
    return np.expand_dims(img, axis=0)

# ═══════════════════════════════════════════════════════════════
#  SKIN MODULE — PyTorch / EfficientNetB3
# ═══════════════════════════════════════════════════════════════
SKIN_MODEL  = None
SKIN_DEVICE = None
SKIN_LOADED = False

SKIN_CLASS_NAMES  = ['anemia', 'cyanosis', 'jaundice', 'normal']
SKIN_IDX_TO_CLASS = {0:'anemia', 1:'cyanosis', 2:'jaundice', 3:'normal'}
SKIN_RISK = {
    "anemia"  : {"level":"MODERATE","color":"#f97316","advice":"Obtain CBC. Check haemoglobin. Assess for haemolysis. Pale skin tone suggests possible anaemia."},
    "cyanosis": {"level":"CRITICAL","color":"#ef4444","advice":"URGENT: Check O₂ saturation immediately. May indicate cardiac or respiratory issue. Escalate now."},
    "jaundice": {"level":"MODERATE","color":"#f59e0b","advice":"Check bilirubin levels. Consider phototherapy if bilirubin >15 mg/dL. Monitor closely."},
    "normal"  : {"level":"NORMAL",  "color":"#22c55e","advice":"Skin tone appears normal. No signs of jaundice, cyanosis or anaemia detected. Routine monitoring."},
}
SKIN_LABELS = {"anemia":"Anaemia","cyanosis":"Cyanosis","jaundice":"Jaundice","normal":"Normal Skin"}
SKIN_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SKIN_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def ensure_skin_loaded():
    global SKIN_MODEL, SKIN_DEVICE, SKIN_LOADED
    if SKIN_LOADED:
        return
    SKIN_LOADED = True
    ensure_file("best_model.pth")
    try:
        import torch, torch.nn as nn
        from torchvision.models import efficientnet_b3
        SKIN_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists("best_model.pth"):
            print("⚠️  Skin model not found — DEMO mode"); return

        class NeoScanSkinModel(nn.Module):
            def __init__(self, num_classes=4):
                super().__init__()
                backbone = efficientnet_b3(weights=None)
                self.features   = backbone.features
                self.avgpool    = backbone.avgpool
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.4, inplace=True),
                    nn.Linear(backbone.classifier[1].in_features, num_classes)
                )
            def forward(self, x):
                x = self.features(x); x = self.avgpool(x)
                return self.classifier(torch.flatten(x, 1))

        model = NeoScanSkinModel(4).to(SKIN_DEVICE)
        ckpt  = torch.load("best_model.pth", map_location=SKIN_DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval(); SKIN_MODEL = model
        print("✅ Skin model loaded")
    except Exception as e:
        print(f"⚠️  Skin load error: {e} — DEMO mode")

def white_balance(img):
    f = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(f[:,:,0]), np.mean(f[:,:,1]), np.mean(f[:,:,2])
    g = (avg_b + avg_g + avg_r) / 3.0
    f[:,:,0] = np.clip(f[:,:,0]*(g/avg_b), 0, 255)
    f[:,:,1] = np.clip(f[:,:,1]*(g/avg_g), 0, 255)
    f[:,:,2] = np.clip(f[:,:,2]*(g/avg_r), 0, 255)
    return f.astype(np.uint8)

def apply_skin_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def preprocess_skin(image_bytes):
    import torch
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not decode image")
    img = white_balance(img)
    img = apply_skin_clahe(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = (img.astype(np.float32)/255.0 - SKIN_MEAN) / SKIN_STD
    return torch.from_numpy(np.transpose(img, (2,0,1))).unsqueeze(0).to(SKIN_DEVICE)

# ═══════════════════════════════════════════════════════════════
#  BODY MODULE — PyTorch / ResNet50
# ═══════════════════════════════════════════════════════════════
BODY_MODEL  = None
BODY_DEVICE = None
BODY_LOADED = False

BODY_RISK   = {
    "normal"  : {"level":"NORMAL","color":"#22c55e","advice":"Body posture appears normal. Limb symmetry detected. Routine monitoring."},
    "abnormal": {"level":"HIGH",  "color":"#ef4444","advice":"Abnormal body posture detected. Possible limb asymmetry, hydrops or restricted movement. Immediate pediatric review required."},
}
BODY_LABELS = {"normal":"Normal Posture","abnormal":"Abnormal Posture"}
BODY_MEAN   = [0.485, 0.456, 0.406]
BODY_STD    = [0.229, 0.224, 0.225]

def ensure_body_loaded():
    global BODY_MODEL, BODY_DEVICE, BODY_LOADED
    if BODY_LOADED:
        return
    BODY_LOADED = True
    ensure_file("body_model.pth")
    try:
        import torch, torch.nn as nn
        from torchvision import models
        BODY_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists("body_model.pth"):
            print("⚠️  Body model not found — DEMO mode"); return

        class NeoScanNet(nn.Module):
            def __init__(self):
                super().__init__()
                base = models.resnet50(weights=None)
                self.backbone = nn.Sequential(*list(base.children())[:-1])
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
                    nn.Linear(512,128),  nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128,2)
                )
            def forward(self, x): return self.head(self.backbone(x))

        model = NeoScanNet().to(BODY_DEVICE)
        ckpt  = torch.load("body_model.pth", map_location=BODY_DEVICE)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval(); BODY_MODEL = model
        print("✅ Body model loaded")
    except Exception as e:
        print(f"⚠️  Body load error: {e} — DEMO mode")

def preprocess_body(image_bytes):
    import torch, torchvision.transforms as T
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not decode image")
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(BODY_MEAN, BODY_STD)])(pil).unsqueeze(0).to(BODY_DEVICE)

# ═══════════════════════════════════════════════════════════════
#  FACE MODULE — PyTorch / VGG16 + ViT Ensemble
# ═══════════════════════════════════════════════════════════════
FACE_VGG    = None
FACE_VIT    = None
FACE_DEVICE = None
FACE_LOADED = False

FACE_CLASS_NAMES = ['Cleft','DownSyndrome','Healthy']
FACE_IDX2LABEL   = {0:'Cleft',1:'DownSyndrome',2:'Healthy'}
FACE_MEAN        = [0.485, 0.456, 0.406]
FACE_STD         = [0.229, 0.224, 0.225]
FACE_RISK = {
    'Cleft'       : {'level':'HIGH',  'color':'#ef4444','advice':'Cleft lip/palate detected. Refer to craniofacial surgery team immediately. Pre-surgical evaluation recommended.'},
    'DownSyndrome': {'level':'HIGH',  'color':'#f97316','advice':'Down Syndrome facial features detected. Refer to genetics team. Cardiac screening and developmental assessment required.'},
    'Healthy'     : {'level':'NORMAL','color':'#22c55e','advice':'No facial abnormalities detected. Normal neonatal facial features. Routine monitoring recommended.'},
}
FACE_LABELS = {'Cleft':'Cleft Lip / Palate','DownSyndrome':'Down Syndrome','Healthy':'Healthy — Normal'}

def ensure_face_loaded():
    global FACE_VGG, FACE_VIT, FACE_DEVICE, FACE_LOADED
    if FACE_LOADED:
        return
    FACE_LOADED = True
    ensure_file("VGG16_best.pth")
    ensure_file("ViT_v2_best.pth")
    try:
        import torch, torch.nn as nn
        from torchvision import models
        FACE_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class VGG16Classifier(nn.Module):
            def __init__(self, n=3):
                super().__init__()
                vgg = models.vgg16(weights=None)
                self.features = vgg.features; self.avgpool = vgg.avgpool
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512*7*7,1024), nn.BatchNorm1d(1024), nn.ReLU(True), nn.Dropout(0.65),
                    nn.Linear(1024,512),     nn.BatchNorm1d(512),  nn.ReLU(True), nn.Dropout(0.5),
                    nn.Linear(512,128),      nn.ReLU(True), nn.Dropout(0.3),
                    nn.Linear(128,n)
                )
            def forward(self, x): x=self.features(x); x=self.avgpool(x); return self.classifier(x)

        if os.path.exists("VGG16_best.pth"):
            vgg = VGG16Classifier(3).to(FACE_DEVICE)
            ckpt = torch.load("VGG16_best.pth", map_location=FACE_DEVICE)
            vgg.load_state_dict(ckpt.get('model_state_dict', ckpt)); vgg.eval(); FACE_VGG = vgg
            print('✅ Face VGG16 loaded')

        try:
            import timm
            class ViTFaceClassifier(nn.Module):
                def __init__(self, n=3, d=0.3):
                    super().__init__()
                    self.vit  = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0, drop_rate=d)
                    self.head = nn.Sequential(
                        nn.LayerNorm(self.vit.embed_dim),
                        nn.Linear(self.vit.embed_dim,256), nn.GELU(), nn.Dropout(d),
                        nn.Linear(256,n)
                    )
                def forward(self, x): return self.head(self.vit(x))

            if os.path.exists("ViT_v2_best.pth"):
                vit = ViTFaceClassifier(3).to(FACE_DEVICE)
                ckpt = torch.load("ViT_v2_best.pth", map_location=FACE_DEVICE)
                vit.load_state_dict(ckpt.get('model_state_dict', ckpt)); vit.eval(); FACE_VIT = vit
                print('✅ Face ViT loaded')
        except ImportError:
            print('⚠️  timm not installed')
    except Exception as e:
        print(f'⚠️  Face load error: {e} — DEMO mode')

def preprocess_face(image_bytes):
    import torch, torchvision.transforms as T
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError('Could not decode image')
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(FACE_MEAN, FACE_STD)])(pil).unsqueeze(0).to(FACE_DEVICE)

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "NeoScan AI API running", "version": "4.0.0"}

@app.get("/health")
def health():
    return {
        "status"           : "ok",
        "eye_model_loaded" : EYE_MODEL  is not None,
        "skin_model_loaded": SKIN_MODEL is not None,
        "body_model_loaded": BODY_MODEL is not None,
        "face_model_loaded": FACE_VGG is not None or FACE_VIT is not None,
        "eye_mode"         : "live" if EYE_MODEL  is not None else "demo",
        "skin_mode"        : "live" if SKIN_MODEL is not None else "demo",
        "body_mode"        : "live" if BODY_MODEL is not None else "demo",
        "face_mode"        : "live" if FACE_VGG is not None or FACE_VIT is not None else "demo",
    }

@app.post("/predict/eye")
async def predict_eye(file: UploadFile = File(...), image_type: str = "rop"):
    if not file.content_type.startswith("image/"): raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()
    ensure_eye_loaded()
    if EYE_MODEL is None:
        import random
        classes=list(EYE_RISK.keys()); pred=random.choice(classes); probs=np.random.dirichlet(np.ones(6)).tolist(); risk=EYE_RISK[pred]
        return {"prediction":pred,"label":EYE_LABELS[pred],"confidence":round(max(probs)*100,2),"risk_level":risk["level"],"risk_color":risk["color"],"advice":risk["advice"],"probabilities":{EYE_LABELS[c]:round(p*100,2) for c,p in zip(classes,probs)},"mode":"demo"}
    try:
        img=preprocess_eye(image_bytes,image_type); probs=EYE_MODEL.predict(img,verbose=0)[0]; pred_idx=int(np.argmax(probs))
        idx_map=EYE_CONFIG.get("idx_to_class",{}); pred_class=idx_map.get(str(pred_idx),"unknown"); risk=EYE_RISK.get(pred_class,EYE_RISK["normal_eye"])
        return {"prediction":pred_class,"label":EYE_LABELS.get(pred_class,pred_class),"confidence":round(float(probs[pred_idx])*100,2),"risk_level":risk["level"],"risk_color":risk["color"],"advice":risk["advice"],"probabilities":{EYE_LABELS.get(idx_map.get(str(i),""),str(i)):round(float(p)*100,2) for i,p in enumerate(probs)},"mode":"live"}
    except Exception as e: raise HTTPException(500, f"Eye prediction error: {str(e)}")

@app.post("/predict/skin")
async def predict_skin(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"): raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()
    ensure_skin_loaded()
    if SKIN_MODEL is None:
        import random
        classes=SKIN_CLASS_NAMES; pred=random.choice(classes); probs=np.random.dirichlet(np.ones(4)).tolist(); risk=SKIN_RISK[pred]
        return {"prediction":pred,"label":SKIN_LABELS[pred],"confidence":round(max(probs)*100,2),"risk_level":risk["level"],"risk_color":risk["color"],"advice":risk["advice"],"probabilities":{SKIN_LABELS[c]:round(p*100,2) for c,p in zip(classes,probs)},"mode":"demo"}
    try:
        import torch
        with torch.no_grad(): probs=torch.softmax(SKIN_MODEL(preprocess_skin(image_bytes)),dim=1).squeeze().cpu().numpy()
        pred_idx=int(np.argmax(probs)); pred_class=SKIN_IDX_TO_CLASS[pred_idx]; risk=SKIN_RISK[pred_class]
        return {"prediction":pred_class,"label":SKIN_LABELS[pred_class],"confidence":round(float(probs[pred_idx])*100,2),"risk_level":risk["level"],"risk_color":risk["color"],"advice":risk["advice"],"probabilities":{SKIN_LABELS[SKIN_IDX_TO_CLASS[i]]:round(float(p)*100,2) for i,p in enumerate(probs)},"mode":"live"}
    except Exception as e: raise HTTPException(500, f"Skin prediction error: {str(e)}")

@app.post("/predict/body")
async def predict_body(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"): raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()
    ensure_body_loaded()
    if BODY_MODEL is None:
        import random
        pred=random.choice(["normal","abnormal"]); probs=np.random.dirichlet(np.ones(2)).tolist(); risk=BODY_RISK[pred]
        return {"prediction":pred,"label":BODY_LABELS[pred],"confidence":round(max(probs)*100,2),"risk_level":risk["level"],"risk_color":risk["color"],"advice":risk["advice"],"probabilities":{"Normal Posture":round(probs[0]*100,2),"Abnormal Posture":round(probs[1]*100,2)},"mode":"demo"}
    try:
        import torch
        with torch.no_grad(): probs=torch.softmax(BODY_MODEL(preprocess_body(image_bytes)),dim=1).squeeze().cpu().numpy()
        pred_idx=int(np.argmax(probs)); pred_class="normal" if pred_idx==0 else "abnormal"; risk=BODY_RISK[pred_class]
        return {"prediction":pred_class,"label":BODY_LABELS[pred_class],"confidence":round(float(probs[pred_idx])*100,2),"risk_level":risk["level"],"risk_color":risk["color"],"advice":risk["advice"],"probabilities":{"Normal Posture":round(float(probs[0])*100,2),"Abnormal Posture":round(float(probs[1])*100,2)},"mode":"live"}
    except Exception as e: raise HTTPException(500, f"Body prediction error: {str(e)}")

@app.post('/predict/face')
async def predict_face(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'): raise HTTPException(400, 'File must be an image')
    image_bytes = await file.read()
    ensure_face_loaded()
    if FACE_VGG is None and FACE_VIT is None:
        import random
        pred=random.choice(FACE_CLASS_NAMES); probs=np.random.dirichlet(np.ones(3)).tolist(); risk=FACE_RISK[pred]
        return {'prediction':pred,'label':FACE_LABELS[pred],'confidence':round(max(probs)*100,2),'risk_level':risk['level'],'risk_color':risk['color'],'advice':risk['advice'],'probabilities':{FACE_LABELS[c]:round(p*100,2) for c,p in zip(FACE_CLASS_NAMES,probs)},'mode':'demo','model_used':'demo'}
    try:
        import torch
        tensor=preprocess_face(image_bytes); pc=None
        with torch.no_grad():
            if FACE_VGG: pc=torch.softmax(FACE_VGG(tensor),dim=1).cpu().numpy()[0]
            if FACE_VIT:
                vp=torch.softmax(FACE_VIT(tensor),dim=1).cpu().numpy()[0]
                pc=0.5*pc+0.5*vp if pc is not None else vp
        pred_idx=int(np.argmax(pc)); pred_class=FACE_IDX2LABEL[pred_idx]; risk=FACE_RISK[pred_class]
        mode='ensemble' if (FACE_VGG and FACE_VIT) else ('vgg16' if FACE_VGG else 'vit')
        return {'prediction':pred_class,'label':FACE_LABELS[pred_class],'confidence':round(float(pc[pred_idx])*100,2),'risk_level':risk['level'],'risk_color':risk['color'],'advice':risk['advice'],'probabilities':{FACE_LABELS[FACE_IDX2LABEL[i]]:round(float(p)*100,2) for i,p in enumerate(pc)},'mode':'live','model_used':mode}
    except Exception as e: raise HTTPException(500, f'Face prediction error: {str(e)}')
