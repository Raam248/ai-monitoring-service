from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import datetime
import subprocess
import json
import asyncio
import base64
import io
import time
from typing import Optional
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

# FER import with fallback
try:
    from fer.fer import FER  # Updated import path
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("⚠️ FER not installed - video emotion detection disabled")

app = FastAPI()

# ===============================
# CORS (allow frontend)
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# CONFIG
# ===============================

HATE_THRESHOLD = 0.5
MIN_AUDIO_SIZE = 5000  # Lowered for better responsiveness
AUDIO_BUFFER_SIZE = 3  # Number of chunks to accumulate before processing

# ===============================
# Thread pool for blocking operations
# ===============================

executor = ThreadPoolExecutor(max_workers=2)

# ===============================
# Load models ONCE at startup
# ===============================

print("⏳ Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded")

print("⏳ Loading hate speech model...")
hate_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None
)
print("✅ Hate speech model loaded")

# Load FER model for emotion detection
emotion_detector = None
if FER_AVAILABLE:
    print("⏳ Loading emotion detection model...")
    try:
        emotion_detector = FER(mtcnn=True)
        print("✅ Emotion detection model loaded")
    except Exception as e:
        print(f"⚠️ Could not load emotion detector: {e}")

# ===============================
# Session state for multimodal fusion
# ===============================

class SessionState:
    """Tracks recent detections for multimodal fusion"""
    def __init__(self):
        self.recent_audio_alerts = []  # List of (timestamp, alert_level, score)
        self.recent_emotions = []  # List of (timestamp, emotion, score)
        self.window_seconds = 30  # Time window for analysis
    
    def add_audio_alert(self, alert_level: str, score: float):
        now = time.time()
        self.recent_audio_alerts.append((now, alert_level, score))
        self._cleanup()
    
    def add_emotion(self, emotion: str, score: float):
        now = time.time()
        self.recent_emotions.append((now, emotion, score))
        self._cleanup()
    
    def _cleanup(self):
        """Remove old entries"""
        cutoff = time.time() - self.window_seconds
        self.recent_audio_alerts = [(t, a, s) for t, a, s in self.recent_audio_alerts if t > cutoff]
        self.recent_emotions = [(t, e, s) for t, e, s in self.recent_emotions if t > cutoff]
    
    def get_fusion_alert(self) -> dict:
        """Compute multimodal fusion alert"""
        self._cleanup()
        
        # Count recent dangerous audio alerts
        danger_count = sum(1 for _, a, _ in self.recent_audio_alerts if a == "danger")
        warning_count = sum(1 for _, a, _ in self.recent_audio_alerts if a == "warning")
        
        # Check for angry emotions
        angry_emotions = [(e, s) for _, e, s in self.recent_emotions if e == "angry"]
        has_angry = len(angry_emotions) > 0
        avg_angry_score = sum(s for _, s in angry_emotions) / len(angry_emotions) if angry_emotions else 0
        
        # Fusion logic
        fusion_level = "safe"
        fusion_reason = ""
        
        if danger_count >= 2:
            fusion_level = "critical"
            fusion_reason = f"Multiple dangerous speech detected ({danger_count}x)"
        elif danger_count >= 1 and has_angry:
            fusion_level = "critical"
            fusion_reason = "Dangerous speech + angry expression detected"
        elif warning_count >= 3:
            fusion_level = "danger"
            fusion_reason = f"Repeated warnings ({warning_count}x in {self.window_seconds}s)"
        elif has_angry and warning_count >= 1:
            fusion_level = "danger"
            fusion_reason = "Warning speech + angry expression"
        elif has_angry and avg_angry_score > 0.7:
            fusion_level = "warning"
            fusion_reason = "Strong angry expression detected"
        
        return {
            "fusion_level": fusion_level,
            "fusion_reason": fusion_reason,
            "stats": {
                "danger_count": danger_count,
                "warning_count": warning_count,
                "has_angry": has_angry,
                "window_seconds": self.window_seconds
            }
        }

# Global session state (in production, use per-user sessions)
session_state = SessionState()

# ===============================
# Violence keywords for boosting
# ===============================

VIOLENCE_KEYWORDS = [
    "kill", "murder", "die", "death", "hit", "beat", "punch", "slap",
    "hurt", "attack", "stab", "shoot", "strangle", "choke", "destroy",
    "rape", "assault", "abuse", "threaten", "harm"
]

THREAT_PHRASES = [
    "i will", "i'm going to", "gonna", "i'll", "watch out",
    "you're dead", "you will regret", "i swear"
]

def check_violence_keywords(text: str) -> tuple[bool, list]:
    """Check for violence keywords and return matches"""
    text_lower = text.lower()
    found = [kw for kw in VIOLENCE_KEYWORDS if kw in text_lower]
    has_threat_phrase = any(phrase in text_lower for phrase in THREAT_PHRASES)
    return (len(found) > 0 or has_threat_phrase), found

# ===============================
# Helper: Convert WEBM → WAV
# ===============================

def convert_webm_to_wav(webm_path: str, wav_path: str) -> bool:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", webm_path,
            "-ar", "16000", "-ac", "1", wav_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0 and os.path.exists(wav_path)

# ===============================
# Helper: Process audio (runs in thread)
# ===============================

def process_audio_sync(audio_bytes: bytes) -> dict:
    """Synchronous audio processing - runs in thread pool"""
    try:
        # Save WEBM
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            webm_path = tmp.name
        
        wav_path = webm_path.replace(".webm", ".wav")
        
        # Convert
        if not convert_webm_to_wav(webm_path, wav_path):
            os.remove(webm_path)
            return {"error": "Audio conversion failed"}
        
        # Transcribe
        result = whisper_model.transcribe(wav_path, fp16=False)
        text = result["text"].strip()
        
        # Cleanup
        os.remove(webm_path)
        os.remove(wav_path)
        
        if not text:
            return {"error": "No speech detected"}
        
        # Hate classification
        hate_result = hate_classifier(text)
        predictions = hate_result[0]
        
        # Get top prediction
        max_pred = max(predictions, key=lambda x: x["score"])
        model_label = max_pred["label"]
        model_score = max_pred["score"]
        
        # Check violence keywords
        has_violence, violence_words = check_violence_keywords(text)
        
        # Compute final score with boosting
        final_score = model_score
        if has_violence:
            final_score = min(1.0, model_score + 0.3)  # Boost by 0.3 if violence detected
        
        # Determine alert level
        if final_score >= 0.7 or (has_violence and model_score >= 0.3):
            alert_level = "danger"
        elif final_score >= HATE_THRESHOLD:
            alert_level = "warning"
        else:
            alert_level = "safe"
        
        return {
            "type": "audio_result",
            "transcription": text,
            "hate_detection": {
                "label": model_label,
                "model_score": round(model_score, 3),
                "final_score": round(final_score, 3),
                "violence_keywords": violence_words,
                "alert_level": alert_level
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}

# ===============================
# WebSocket: Audio streaming
# ===============================

@app.websocket("/ws/audio")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    print("🔌 Audio WebSocket connected")
    
    audio_buffer = bytearray()
    chunk_count = 0
    
    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            
            if len(audio_bytes) < MIN_AUDIO_SIZE:
                continue
            
            # Accumulate chunks
            audio_buffer.extend(audio_bytes)
            chunk_count += 1
            
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            size_kb = len(audio_buffer) / 1024
            print(f"[{ts}] Buffer: {size_kb:.1f} KB ({chunk_count} chunks)")
            
            # Process when we have enough data (or every chunk for responsiveness)
            if chunk_count >= AUDIO_BUFFER_SIZE or len(audio_buffer) > 50000:
                # Send status
                await ws.send_json({"type": "status", "message": "Processing audio..."})
                
                # Process in thread pool (non-blocking)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor, 
                    process_audio_sync, 
                    bytes(audio_buffer)
                )
                
                # Send result to frontend
                if "error" not in result:
                    await ws.send_json(result)
                    print(f"🗣️ [{ts}] {result['transcription']}")
                    print(f"   Alert: {result['hate_detection']['alert_level']} (score: {result['hate_detection']['final_score']})")
                else:
                    print(f"⚠️ [{ts}] {result['error']}")
                
                # Reset buffer
                audio_buffer = bytearray()
                chunk_count = 0
                
    except WebSocketDisconnect:
        print("🔌 Audio WebSocket disconnected")
    except Exception as e:
        print(f"❌ Audio WebSocket error: {e}")


# ===============================
# Health check endpoint
# ===============================

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": True}


# ===============================
# TEST ENDPOINT
# ===============================

@app.get("/test-hate")
def test_hate(text: str = "I hate you and your stupid ideas"):
    result = hate_classifier(text)
    predictions = result[0]
    max_pred = max(predictions, key=lambda x: x["score"])
    
    has_violence, violence_words = check_violence_keywords(text)
    
    return {
        "input_text": text,
        "top_label": max_pred["label"],
        "confidence": round(max_pred["score"], 3),
        "violence_detected": has_violence,
        "violence_words": violence_words,
        "all_scores": predictions
    }


# ===============================
# Helper: Process video frame
# ===============================

def process_video_frame_sync(image_data: bytes) -> dict:
    """Process a video frame for emotion detection"""
    if not emotion_detector:
        return {"error": "Emotion detector not available"}
    
    try:
        # Decode base64 image
        image = Image.open(io.BytesIO(image_data))
        frame = np.array(image)
        
        # Convert RGB to BGR for OpenCV (FER expects BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]
        
        # Detect emotions
        result = emotion_detector.detect_emotions(frame)
        
        if not result:
            return {"error": "No face detected"}
        
        # Get dominant emotion from first face
        emotions = result[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        dominant_score = emotions[dominant_emotion]
        
        # Add to session state
        session_state.add_emotion(dominant_emotion, dominant_score)
        
        # Get fusion alert
        fusion = session_state.get_fusion_alert()
        
        return {
            "type": "video_result",
            "emotion": {
                "dominant": dominant_emotion,
                "score": round(dominant_score, 3),
                "all_emotions": {k: round(v, 3) for k, v in emotions.items()}
            },
            "fusion": fusion,
            "faces_detected": len(result),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}


# ===============================
# WebSocket: Video streaming
# ===============================

@app.websocket("/ws/video")
async def video_stream(ws: WebSocket):
    await ws.accept()
    print("📹 Video WebSocket connected")
    
    frame_count = 0
    
    try:
        while True:
            # Receive base64 encoded frame
            data = await ws.receive_text()
            
            try:
                # Parse JSON message
                msg = json.loads(data)
                if msg.get("type") != "video_frame":
                    continue
                
                # Decode base64 image data
                image_data = base64.b64decode(msg["data"])
                
                frame_count += 1
                
                # Process every frame (throttle on frontend)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor,
                    process_video_frame_sync,
                    image_data
                )
                
                if "error" not in result:
                    await ws.send_json(result)
                    emotion = result["emotion"]["dominant"]
                    score = result["emotion"]["score"]
                    print(f"📹 Frame {frame_count}: {emotion} ({score:.2f})")
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                
    except WebSocketDisconnect:
        print("📹 Video WebSocket disconnected")
    except Exception as e:
        print(f"❌ Video WebSocket error: {e}")


# ===============================
# Combined WebSocket: Audio + Video
# ===============================

@app.websocket("/ws/monitor")
async def combined_monitor(ws: WebSocket):
    """Single WebSocket for both audio and video"""
    await ws.accept()
    print("🎬 Combined monitor WebSocket connected")
    
    audio_buffer = bytearray()
    chunk_count = 0
    
    try:
        while True:
            # Receive message (can be binary audio or text JSON for video)
            message = await ws.receive()
            
            if "bytes" in message:
                # Audio data
                audio_bytes = message["bytes"]
                
                if len(audio_bytes) < MIN_AUDIO_SIZE:
                    continue
                
                audio_buffer.extend(audio_bytes)
                chunk_count += 1
                
                if chunk_count >= AUDIO_BUFFER_SIZE or len(audio_buffer) > 50000:
                    await ws.send_json({"type": "status", "message": "Processing audio..."})
                    
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        executor,
                        process_audio_sync,
                        bytes(audio_buffer)
                    )
                    
                    if "error" not in result:
                        # Add to session state
                        alert_level = result["hate_detection"]["alert_level"]
                        score = result["hate_detection"]["final_score"]
                        session_state.add_audio_alert(alert_level, score)
                        
                        # Add fusion data
                        result["fusion"] = session_state.get_fusion_alert()
                        await ws.send_json(result)
                    
                    audio_buffer = bytearray()
                    chunk_count = 0
                    
            elif "text" in message:
                # Video frame (JSON)
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "video_frame":
                        image_data = base64.b64decode(msg["data"])
                        
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            executor,
                            process_video_frame_sync,
                            image_data
                        )
                        
                        if "error" not in result:
                            await ws.send_json(result)
                            
                except json.JSONDecodeError:
                    pass
                    
    except WebSocketDisconnect:
        print("🎬 Combined monitor WebSocket disconnected")
    except Exception as e:
        print(f"❌ Combined monitor error: {e}")
