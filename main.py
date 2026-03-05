"""
AI Meeting Monitor - Production Backend
Real-time multimodal hate speech and emotion detection.
"""
import asyncio
import base64
import datetime
import io
import logging
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import config

# ===============================
# Room & User Management
# ===============================

class Room:
    """Represents a meeting room."""
    def __init__(self, room_id: str):
        self.id = room_id
        self.participants = {}  # user_id -> {name, isHost, ws, strikes}
        self.chat_connections = {}  # user_id -> ws
        self.created_at = time.time()
    
    def add_participant(self, user_id: str, name: str, is_host: bool, ws: WebSocket):
        self.participants[user_id] = {
            "id": user_id,
            "name": name,
            "isHost": is_host,
            "ws": ws,
            "strikes": 0
        }
    
    def remove_participant(self, user_id: str):
        self.participants.pop(user_id, None)
        self.chat_connections.pop(user_id, None)
    
    def get_participant_list(self):
        return [{"id": p["id"], "name": p["name"], "isHost": p["isHost"]} 
                for p in self.participants.values()]
    
    def add_strike(self, user_id: str) -> int:
        if user_id in self.participants:
            self.participants[user_id]["strikes"] += 1
            return self.participants[user_id]["strikes"]
        return 0
    
    def get_strikes(self, user_id: str) -> int:
        return self.participants.get(user_id, {}).get("strikes", 0)
    
    async def broadcast(self, message: dict, exclude_user: str = None):
        for user_id, participant in self.participants.items():
            if user_id != exclude_user and participant["ws"]:
                try:
                    await participant["ws"].send_json(message)
                except:
                    pass
    
    async def broadcast_chat(self, message: dict, exclude_user: str = None):
        for user_id, ws in self.chat_connections.items():
            if user_id != exclude_user:
                try:
                    await ws.send_json(message)
                except:
                    pass


class RoomManager:
    """Manages all active rooms."""
    def __init__(self):
        self.rooms: dict[str, Room] = {}
    
    def get_or_create(self, room_id: str) -> Room:
        if room_id not in self.rooms:
            self.rooms[room_id] = Room(room_id)
        return self.rooms[room_id]
    
    def get(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)
    
    def cleanup_empty(self):
        empty = [rid for rid, room in self.rooms.items() if not room.participants]
        for rid in empty:
            del self.rooms[rid]


room_manager = RoomManager()

# ===============================
# Logging Setup
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ===============================
# Model Loading (Lazy)
# ===============================

class ModelManager:
    """Manages ML model lifecycle."""
    
    def __init__(self):
        self.whisper_model = None
        self.hate_classifier = None
        self.emotion_detector = None
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def load_whisper(self):
        """Load Whisper speech-to-text model."""
        if self.whisper_model is None:
            import whisper
            logger.info(f"Loading Whisper model ({config.WHISPER_MODEL})...")
            self.whisper_model = whisper.load_model(config.WHISPER_MODEL)
            logger.info("Whisper model loaded")
        return self.whisper_model
    
    def load_hate_classifier(self):
        """Load hate speech classification model."""
        if self.hate_classifier is None:
            from transformers import pipeline
            import torch
            logger.info("Loading hate speech classifier...")
            # Force PyTorch framework to avoid TensorFlow issues
            self.hate_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=None,
                framework="pt",  # PyTorch only
                device="cpu"
            )
            logger.info("Hate speech classifier loaded")
        return self.hate_classifier
    
    def load_emotion_detector(self):
        """Load facial emotion detector (optional)."""
        if self.emotion_detector is None and config.ENABLE_VIDEO:
            try:
                from fer.fer import FER
                logger.info("Loading emotion detector...")
                self.emotion_detector = FER(mtcnn=True)
                logger.info("Emotion detector loaded")
            except ImportError:
                logger.warning("FER not available - video emotion detection disabled")
            except Exception as e:
                logger.warning(f"Could not load emotion detector: {e}")
        return self.emotion_detector
    
    def get_status(self) -> dict:
        """Get status of all models."""
        return {
            "whisper": self.whisper_model is not None,
            "hate_classifier": self.hate_classifier is not None,
            "emotion_detector": self.emotion_detector is not None,
            "video_enabled": config.ENABLE_VIDEO,
        }
    
    @property
    def executor(self):
        return self._executor


models = ModelManager()

# ===============================
# Session State for Fusion
# ===============================

class SessionState:
    """Tracks recent detections for multimodal fusion."""
    
    def __init__(self):
        self.audio_alerts = []  # (timestamp, level, score)
        self.emotions = []  # (timestamp, emotion, score)
    
    def add_audio_alert(self, level: str, score: float):
        self.audio_alerts.append((time.time(), level, score))
        self._cleanup()
    
    def add_emotion(self, emotion: str, score: float):
        self.emotions.append((time.time(), emotion, score))
        self._cleanup()
    
    def _cleanup(self):
        cutoff = time.time() - config.FUSION_WINDOW_SECONDS
        self.audio_alerts = [x for x in self.audio_alerts if x[0] > cutoff]
        self.emotions = [x for x in self.emotions if x[0] > cutoff]
    
    def compute_fusion(self) -> dict:
        self._cleanup()
        
        danger_count = sum(1 for _, l, _ in self.audio_alerts if l == "danger")
        warning_count = sum(1 for _, l, _ in self.audio_alerts if l == "warning")
        angry_emotions = [s for _, e, s in self.emotions if e == "angry"]
        has_angry = len(angry_emotions) > 0
        
        level, reason = "safe", ""
        
        if danger_count >= 2:
            level, reason = "critical", f"Multiple dangerous speech ({danger_count}x)"
        elif danger_count >= 1 and has_angry:
            level, reason = "critical", "Dangerous speech + angry expression"
        elif warning_count >= 3:
            level, reason = "danger", f"Repeated warnings ({warning_count}x)"
        elif has_angry and warning_count >= 1:
            level, reason = "danger", "Warning + angry expression"
        elif has_angry and max(angry_emotions, default=0) > 0.7:
            level, reason = "warning", "Strong angry expression"
        
        return {"level": level, "reason": reason}


session = SessionState()

# ===============================
# Detection Logic
# ===============================

VIOLENCE_KEYWORDS = {
    "kill", "murder", "die", "death", "hit", "beat", "punch", "slap",
    "hurt", "attack", "stab", "shoot", "strangle", "choke", "destroy",
    "rape", "assault", "abuse", "threaten", "harm"
}

THREAT_PHRASES = [
    "i will", "i'm going to", "gonna", "i'll", "watch out",
    "you're dead", "you will regret", "i swear"
]


def detect_violence_keywords(text: str) -> tuple[bool, list]:
    text_lower = text.lower()
    found = [kw for kw in VIOLENCE_KEYWORDS if kw in text_lower]
    has_threat = any(p in text_lower for p in THREAT_PHRASES)
    return (bool(found) or has_threat), found


def convert_audio(webm_path: str, wav_path: str) -> bool:
    """Convert WebM audio to WAV format."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True
    )
    return result.returncode == 0 and os.path.exists(wav_path)


def process_audio(audio_bytes: bytes) -> dict:
    """Process audio chunk - runs in thread pool."""
    start_time = time.time()
    webm_path = wav_path = None
    try:
        # Save and convert
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
            f.write(audio_bytes)
            webm_path = f.name
        wav_path = webm_path.replace(".webm", ".wav")
        
        if not convert_audio(webm_path, wav_path):
            return {"error": "Audio conversion failed"}
        
        # Transcribe
        result = models.whisper_model.transcribe(wav_path, fp16=False)
        text = result["text"].strip()
        
        if not text:
            return {"error": "No speech detected"}
        
        # Classify
        predictions = models.hate_classifier(text)[0]
        top = max(predictions, key=lambda x: x["score"])
        
        # Check violence keywords
        has_violence, keywords = detect_violence_keywords(text)
        
        # Compute final score
        score = min(1.0, top["score"] + (0.3 if has_violence else 0))
        
        # Determine alert level
        if score >= config.DANGER_THRESHOLD or (has_violence and top["score"] >= 0.3):
            level = "danger"
        elif score >= config.HATE_THRESHOLD:
            level = "warning"
        else:
            level = "safe"
        
        # Update session
        session.add_audio_alert(level, score)
        
        processing_time = round(time.time() - start_time, 2)
        return {
            "type": "audio_result",
            "transcription": text,
            "detection": {
                "label": top["label"],
                "score": round(score, 3),
                "level": level,
                "keywords": keywords
            },
            "fusion": session.compute_fusion(),
            "processing_time_sec": processing_time,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return {"error": str(e)}
    finally:
        for path in [webm_path, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)


def process_video_frame(image_bytes: bytes) -> dict:
    """Process video frame for emotion detection."""
    if not models.emotion_detector:
        return {"error": "Emotion detection not available"}
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
        
        # RGB to BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]
        
        results = models.emotion_detector.detect_emotions(frame)
        
        if not results:
            return {"error": "No face detected"}
        
        emotions = results[0]["emotions"]
        dominant = max(emotions, key=emotions.get)
        score = emotions[dominant]
        
        session.add_emotion(dominant, score)
        
        return {
            "type": "video_result",
            "emotion": {
                "dominant": dominant,
                "score": round(score, 3),
                "all": {k: round(v, 3) for k, v in emotions.items()}
            },
            "fusion": session.compute_fusion(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return {"error": str(e)}


# ===============================
# FastAPI App
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Starting AI Meeting Monitor...")
    models.load_whisper()
    models.load_hate_classifier()
    if config.ENABLE_VIDEO:
        models.load_emotion_detector()
    logger.info("All models loaded - server ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Meeting Monitor",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check with component status."""
    return {
        "status": "ok",
        "models": models.get_status(),
        "config": {
            "video_enabled": config.ENABLE_VIDEO,
            "whisper_model": config.WHISPER_MODEL,
        }
    }


@app.get("/test")
def test_detection(text: str = "I hate you"):
    """Test hate detection on text."""
    predictions = models.hate_classifier(text)[0]
    top = max(predictions, key=lambda x: x["score"])
    has_violence, keywords = detect_violence_keywords(text)
    
    return {
        "text": text,
        "label": top["label"],
        "score": round(top["score"], 3),
        "violence_keywords": keywords,
        "predictions": predictions
    }


# Track aggressive emotion counts per user for video moderation
user_aggression_counts: dict[str, int] = {}
AGGRESSION_THRESHOLD = 3  # Consecutive aggressive frames to trigger warning
AGGRESSIVE_EMOTIONS = {"angry": 0.6, "disgust": 0.7, "fear": 0.8}  # emotion: min_score


@app.websocket("/ws/monitor")
async def monitor_websocket(ws: WebSocket):
    """Combined audio + video monitoring WebSocket."""
    await ws.accept()
    logger.info("Client connected")
    
    audio_buffer = bytearray()
    chunk_count = 0
    current_user_id = None
    
    try:
        while True:
            message = await ws.receive()
            
            # Audio (binary)
            if "bytes" in message:
                data = message["bytes"]
                if len(data) < config.MIN_AUDIO_SIZE:
                    continue
                
                audio_buffer.extend(data)
                chunk_count += 1
                
                if chunk_count >= config.AUDIO_BUFFER_CHUNKS:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        models.executor,
                        process_audio,
                        bytes(audio_buffer)
                    )
                    
                    if "error" not in result:
                        await ws.send_json(result)
                        logger.info(f"Transcribed: {result['transcription'][:50]}...")
                        
                        # Check for audio violation
                        if result.get("detection", {}).get("level") in ["danger", "warning"]:
                            await ws.send_json({
                                "type": "violation",
                                "source": "audio",
                                "reason": f"Detected: {result['detection']['label']} in speech",
                                "severity": "high" if result["detection"]["level"] == "danger" else "medium",
                                "transcription": result["transcription"][:100]
                            })
                    
                    audio_buffer.clear()
                    chunk_count = 0
            
            # Video (JSON)
            elif "text" in message:
                import json
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "video_frame":
                        current_user_id = msg.get("userId")
                        user_name = msg.get("userName", "Unknown")
                        
                        if models.emotion_detector:
                            image_data = base64.b64decode(msg["data"])
                            
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                models.executor,
                                process_video_frame,
                                image_data
                            )
                            
                            if "error" not in result:
                                await ws.send_json(result)
                                
                                # Check for aggressive emotions
                                emotion = result.get("emotion", {})
                                dominant = emotion.get("dominant", "")
                                score = emotion.get("score", 0)
                                
                                is_aggressive = False
                                for agg_emotion, min_score in AGGRESSIVE_EMOTIONS.items():
                                    if dominant == agg_emotion and score >= min_score:
                                        is_aggressive = True
                                        break
                                
                                if current_user_id:
                                    if is_aggressive:
                                        user_aggression_counts[current_user_id] = user_aggression_counts.get(current_user_id, 0) + 1
                                        
                                        if user_aggression_counts[current_user_id] >= AGGRESSION_THRESHOLD:
                                            logger.warning(f"Video violation: {user_name} showing sustained {dominant} expression")
                                            await ws.send_json({
                                                "type": "violation",
                                                "source": "video",
                                                "reason": f"Aggressive behavior detected ({dominant} expression)",
                                                "severity": "medium",
                                                "emotion": dominant,
                                                "confidence": round(score, 2)
                                            })
                                            user_aggression_counts[current_user_id] = 0  # Reset after warning
                                    else:
                                        # Reset count on non-aggressive frame
                                        user_aggression_counts[current_user_id] = max(0, user_aggression_counts.get(current_user_id, 0) - 1)
                except:
                    pass
                    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        if current_user_id and current_user_id in user_aggression_counts:
            del user_aggression_counts[current_user_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.websocket("/ws/room/{room_id}")
async def room_websocket(ws: WebSocket, room_id: str):
    """Room signaling WebSocket for join/leave and participant management."""
    await ws.accept()
    room = room_manager.get_or_create(room_id)
    current_user_id = None
    
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "join":
                current_user_id = data.get("userId")
                user_name = data.get("userName", "Anonymous")
                is_host = data.get("isHost", False)
                
                room.add_participant(current_user_id, user_name, is_host, ws)
                logger.info(f"User {user_name} joined room {room_id}")
                
                # Send current room state to new user
                await ws.send_json({
                    "type": "room_state",
                    "participants": room.get_participant_list()
                })
                
                # Broadcast join to others
                await room.broadcast({
                    "type": "user_joined",
                    "userId": current_user_id,
                    "userName": user_name,
                    "isHost": is_host
                }, exclude_user=current_user_id)
            
            elif msg_type == "moderation_alert":
                # Broadcast moderation alert to all participants
                await room.broadcast({
                    "type": "moderation_alert",
                    "userId": data.get("userId"),
                    "userName": data.get("userName"),
                    "reason": data.get("reason"),
                    "strike": data.get("strike")
                })
            
            elif msg_type == "kick_user" and current_user_id:
                # Host can kick users
                host_info = room.participants.get(current_user_id, {})
                if host_info.get("isHost"):
                    kick_user_id = data.get("targetUserId")
                    await room.broadcast({
                        "type": "user_kicked",
                        "userId": kick_user_id,
                        "reason": data.get("reason", "Removed by host")
                    })
                    room.remove_participant(kick_user_id)
    
    except WebSocketDisconnect:
        if current_user_id:
            room.remove_participant(current_user_id)
            await room.broadcast({
                "type": "user_left",
                "userId": current_user_id
            })
            logger.info(f"User {current_user_id} left room {room_id}")
        room_manager.cleanup_empty()
    except Exception as e:
        logger.error(f"Room WebSocket error: {e}")


@app.websocket("/ws/chat/{room_id}")
async def chat_websocket(ws: WebSocket, room_id: str):
    """Chat WebSocket with AI moderation for text messages."""
    await ws.accept()
    room = room_manager.get_or_create(room_id)
    current_user_id = None
    
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "chat_message":
                current_user_id = data.get("userId")
                user_name = data.get("userName", "Anonymous")
                text = data.get("text", "")
                
                # Register chat connection
                if current_user_id:
                    room.chat_connections[current_user_id] = ws
                
                # AI moderation on chat text
                if text.strip():
                    try:
                        predictions = models.hate_classifier(text)[0]
                        top = max(predictions, key=lambda x: x["score"])
                        has_violence, keywords = detect_violence_keywords(text)
                        score = min(1.0, top["score"] + (0.3 if has_violence else 0))
                        
                        # Check if message violates policy
                        if score >= config.HATE_THRESHOLD or has_violence:
                            strikes = room.add_strike(current_user_id)
                            severity = "high" if score >= config.DANGER_THRESHOLD else "medium"
                            
                            # Send warning to offender
                            await ws.send_json({
                                "type": "moderation_warning",
                                "reason": f"Message flagged: {top['label']}",
                                "severity": severity,
                                "strike": strikes,
                                "message": text[:50] + "..." if len(text) > 50 else text
                            })
                            
                            # Broadcast warning to room
                            await room.broadcast_chat({
                                "type": "moderation_warning",
                                "userName": user_name,
                                "reason": "Message contained prohibited content",
                                "strike": strikes
                            }, exclude_user=current_user_id)
                            
                            logger.warning(f"Chat violation by {user_name}: {text[:30]}... (strike {strikes})")
                            
                            # Auto-kick at 3 strikes
                            if strikes >= 3:
                                await room.broadcast({
                                    "type": "user_kicked",
                                    "userId": current_user_id,
                                    "reason": "Exceeded strike limit (3 violations)"
                                })
                                room.remove_participant(current_user_id)
                                logger.info(f"User {user_name} kicked for 3 strikes")
                            
                            continue  # Don't broadcast the violating message
                    except Exception as e:
                        logger.error(f"Chat moderation error: {e}")
                
                # Broadcast clean message to others
                await room.broadcast_chat({
                    "type": "chat_message",
                    "userId": current_user_id,
                    "userName": user_name,
                    "text": text
                }, exclude_user=current_user_id)
    
    except WebSocketDisconnect:
        if current_user_id:
            room.chat_connections.pop(current_user_id, None)
        logger.info(f"Chat disconnected for room {room_id}")
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )
