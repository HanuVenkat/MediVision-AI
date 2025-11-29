# backend/brain_of_the_doctor.py
import os
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from .database import SessionLocal, engine, Base
from .models import SkinRecord
import speech_recognition as sr
from gtts import gTTS

# audio helpers
from io import BytesIO
from pydub import AudioSegment

# ML / image imports
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# ---------- CONFIG & FILE PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- DATABASE (ensure tables exist) ----------
Base.metadata.create_all(bind=engine)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# ---------- DEVICE & MODEL (CPU) ----------
device = torch.device("cpu")
model = models.mobilenet_v2(pretrained=True).to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

app = FastAPI(title="Medivision AI - Backend")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Improved classifier ----------
def ml_skin_classifier(image_path):
    """
    Improved rule order:
      - Check dark spots first (moles / hyperpigmentation)
      - Then check redness (acne)
      - Use feature score as fallback
    Returns: "acne_inflammation", "hyperpigmentation_or_moles", "other_or_normal"
    """
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.uint8)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # brightness
    brightness = (r.astype(int) + g.astype(int) + b.astype(int)) / 3.0

    # Dark-spot detection (mole / hyperpigmentation)
    dark_mask = brightness < 60
    dark_ratio = dark_mask.sum() / (224 * 224)

    # Redness detection (acne) — conservative thresholds
    red_mask = (r > 130) & (r > g + 25) & (r > b + 25)
    red_ratio = red_mask.sum() / (224 * 224)

    # Optional HSV red enhancement (if cv2 available)
    try:
        import cv2
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red_mask_hsv = ((h <= 10) | (h >= 170)) & (s > 70) & (v > 50)
        red_ratio = max(red_ratio, red_mask_hsv.sum() / (224 * 224))
    except Exception:
        pass

    # Model feature score (fallback)
    try:
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_tensor)
            feat_score = float(out.abs().mean().cpu().numpy())
    except Exception:
        feat_score = 0.0

    # Decision tree (dark first)
    if dark_ratio >= 0.06:
        return "hyperpigmentation_or_moles"
    if red_ratio >= 0.12 or (feat_score > 0.20 and red_ratio >= 0.04):
        return "acne_inflammation"
    return "other_or_normal"

# ---------- Helpers ----------
def ensure_wav(audio_path: str) -> str:
    return audio_path

def recognize_voice(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return None
    except Exception as exc:
        raise RuntimeError(f"Speech recognition error: {exc}")

def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------- /analyze endpoint ----------
@app.post("/analyze/")
async def analyze(image: UploadFile = File(...), audio: UploadFile = File(...)):
    """
    Accepts image + audio files (uploaded), transcribes voice, classifies image,
    builds structured response text, generates WAV TTS, stores an optional DB record,
    and returns JSON with audio_url.
    """
    try:
        ts = int(time.time())

        # save uploaded image
        image_fname = f"upload_{ts}_{image.filename}"
        image_path = str(UPLOADS_DIR / image_fname)
        with open(image_path, "wb") as f:
            f.write(await image.read())

        # save uploaded audio
        audio_fname_in = f"upload_{ts}_{audio.filename}"
        audio_path = str(UPLOADS_DIR / audio_fname_in)
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        # transcribe audio (best-effort)
        try:
            question_text = recognize_voice(audio_path)
            if not question_text:
                raise HTTPException(status_code=400, detail="Could not understand audio. Use clear WAV/MP3.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Speech recognition error: {e}")

        # ---------- Image classification with debugging info ----------
        try:
            label = ml_skin_classifier(image_path)
        except Exception as e:
            label = "other_or_normal"
            print("Classifier exception:", e)

        # Debug prints (will appear in backend terminal)
        try:
            img_dbg = Image.open(image_path).convert("RGB").resize((224, 224))
            arr_dbg = np.array(img_dbg).astype(np.uint8)
            r_dbg, g_dbg, b_dbg = arr_dbg[:, :, 0], arr_dbg[:, :, 1], arr_dbg[:, :, 2]
            red_mask_dbg = (r_dbg > 120) & (r_dbg > g_dbg + 20) & (r_dbg > b_dbg + 20)
            red_ratio_dbg = red_mask_dbg.sum() / (224 * 224)
            brightness_dbg = (r_dbg.astype(int) + g_dbg.astype(int) + b_dbg.astype(int)) / 3.0
            dark_mask_dbg = brightness_dbg < 60
            dark_ratio_dbg = dark_mask_dbg.sum() / (224 * 224)
            feat_score_dbg = 0.0
            try:
                input_tensor_dbg = preprocess(img_dbg).unsqueeze(0).to(device)
                with torch.no_grad():
                    out_dbg = model(input_tensor_dbg)
                    feat_score_dbg = float(out_dbg.abs().mean().cpu().numpy())
            except Exception:
                feat_score_dbg = 0.0

            print(f"[DEBUG] red_ratio={red_ratio_dbg:.4f}, dark_ratio={dark_ratio_dbg:.4f}, feat_score={feat_score_dbg:.4f}, label={label}")
        except Exception as ex_dbg:
            print("Debug print error:", ex_dbg)

        # ---------- Structured suggestions ----------
        RESPONSE_MAP = {
            "acne_inflammation": {
                "title": "Inflammatory acne detected",
                "summary": "The image shows signs of inflamed acne (red bumps).",
                "suggestions": [
                    "Cleanse gently twice daily using a mild, non-comedogenic cleanser,Neutrogena Oil-Free Acne Wash.",
                    "Avoid picking/popping pimples to reduce scarring and infection.",
                    "Use oil-free moisturizers to prevent dryness.",
                    "Consider OTC benzoyl peroxide or salicylic acid after patch-testing."
                ],
                "precautions": "Stop any new product if severe irritation occurs.",
                "when_to_see_doctor": "See a dermatologist if painful cysts or no improvement after several weeks."
            },
            "hyperpigmentation_or_moles": {
                "title": "Dark spots / mole-like patches detected",
                "summary": "The image shows darker patches or mole-like spots.",
                "suggestions": [
                    "Use broad-spectrum sunscreen (SPF 30+) daily,Glenmark Demelan Cream 20g.",
                    "Avoid aggressive scrubs and harsh chemical peels at home.",
                    "Consider gentle brightening ingredients (vitamin C, niacinamide) after consulting a professional."
                ],
                "precautions": "Do not self-treat suspicious or changing moles.",
                "when_to_see_doctor": "See a dermatologist promptly for changing moles (shape, color, borders, bleeding)."
            },
            "other_or_normal": {
                "title": "No specific abnormality detected",
                "summary": "No clear inflammatory acne or suspicious pigmented lesion detected.",
                "suggestions": [
                    "Maintain gentle cleansing and moisturizing.",
                    "Use sunscreen daily.",
                    "Seek dermatology advice if new symptoms appear."
                ],
                "precautions": "",
                "when_to_see_doctor": "If symptoms start or the condition worsens, consult a dermatologist."
            }
        }

        entry = RESPONSE_MAP.get(label, RESPONSE_MAP["other_or_normal"])
        response_lines = [entry["title"], entry["summary"], "", "Suggestions:"]
        response_lines += [f"- {s}" for s in entry["suggestions"]]
        if entry.get("precautions"):
            response_lines += ["", "Precautions:", entry["precautions"]]
        response_lines += ["", "When to see a doctor:", entry["when_to_see_doctor"]]
        response_text = " ".join(response_lines)

        # ---------- TTS: gTTS -> in-memory MP3 -> convert to WAV (pydub) ----------
        try:
            mp3_temp = BytesIO()
            tts = gTTS(text=response_text)
            tts.write_to_fp(mp3_temp)
            mp3_temp.seek(0)

            audio_fname = f"response_{ts}.wav"
            audio_out_path = str(OUTPUT_DIR / audio_fname)

            # Convert MP3 -> WAV (requires ffmpeg in PATH)
            sound = AudioSegment.from_file(mp3_temp, format="mp3")
            sound.export(audio_out_path, format="wav")
        except Exception as tts_exc:
            print("TTS/Conversion error:", tts_exc)
            # create a tiny silent WAV placeholder so front-end still receives a file
            import wave, struct
            audio_fname = f"response_{ts}.wav"
            audio_out_path = str(OUTPUT_DIR / audio_fname)
            with wave.open(audio_out_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 1600)

        # ---------- Save DB record (best-effort) ----------
        record_id = None
        try:
            db = next(get_db())
            record = SkinRecord(
                image_name=os.path.basename(image_path),
                question=question_text,
                ai_response=response_text,
                audio_path=os.path.relpath(audio_out_path, start=BASE_DIR)
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            record_id = record.id
        except Exception:
            record_id = None

        return JSONResponse({
            "id": record_id,
            "label": label,
            "question": question_text,
            "response_text": response_text,
            "audio_url": f"/get_audio/{audio_fname}"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- GET AUDIO ----------
@app.get("/get_audio/{filename}")
def get_audio(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    # WAV file, serve as audio/wav or audio/mpeg if you used mp3
    return FileResponse(str(path), media_type="audio/wav")

# ---------- LIST RECORDS ----------
@app.get("/records/")
def list_records():
    db = next(get_db())
    rows = db.query(SkinRecord).all()
    result = []
    for r in rows:
        result.append({
            'id': r.id,
            'image_name': r.image_name,
            'question': r.question,
            'ai_response': r.ai_response,
            'audio_path': r.audio_path
        })
    return result

# ---------- RUNNER ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.brain_of_the_doctor:app", host="127.0.0.1", port=8000, reload=True)
