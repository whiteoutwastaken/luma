"""
server.py
──────────
Flask server that bridges luma.html and the Whisper Python services.

Run:
    python server.py

Then open:
    http://localhost:5000
"""

import asyncio
import io
import os
import sys
import threading
import webbrowser

import edge_tts
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "services"))
sys.path.append(os.path.join(ROOT, "core"))

from dotenv import load_dotenv
load_dotenv()

from featherless_service import FeatherlessService
from opennote_service    import OpenNoteService
from lesson_manager      import LessonManager, State
from facial_monitor      import FacialMonitor
from note_saver          import NoteSaver
from stt_service         import STTService

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend")
CORS(app)

# ── Face state dict (defined early so update_face_state is available at startup)
_face_state = {"state": "unknown", "confidence": 0.0, "note": ""}

# ── Services (initialised once at startup) ────────────────────────────────────
print("[Server] Initialising services...")
featherless = FeatherlessService()
opennote    = OpenNoteService(featherless=featherless)
note_saver  = NoteSaver(featherless)
stt         = STTService(model="base")


class MockTTS:
    """Stub — real TTS is streamed to the browser via /api/speak."""
    def speak(self, text, blocking=True):
        pass


manager = LessonManager(opennote, featherless, MockTTS(), note_saver)
# Pass lesson_manager=None so the monitor analyses every frame, not just
# during WAITING state. Signals are forwarded to the manager manually
# via the on_state_detected callback below.
monitor = FacialMonitor(featherless, lesson_manager=None)
def _on_face_detected(state, conf):
    _face_state.update({'state': state, 'confidence': conf, 'note': ''})
    # Forward actionable signals to the lesson manager
    if state not in ('focused', 'unknown', 'engaged'):
        manager.signal(state)

monitor.on_state_detected = _on_face_detected
monitor.on_no_face = lambda: _face_state.update({'state': 'no face', 'confidence': 0.0, 'note': ''})
print("[Server] Ready. Camera monitor starts when you click Start Camera in the UI.")


# ── Serve UI ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend", "luma.html")


# ── Lesson generation ─────────────────────────────────────────────────────────
@app.route("/api/lesson", methods=["POST"])
def lesson():
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    try:
        l = opennote.generate_lesson(topic)
        l = featherless.enrich_lesson(l)
        l = featherless.chunk_lesson(l)
        return jsonify({
            "topic":        l.topic,
            "chunks":       l.chunks,
            "images":       [],
            "queries":      l.image_queries,
            "summaries":    l.source_summaries,
            "video_url":    l.video_url,
            "from_journal": l.from_journal,
        })
    except Exception as e:
        print(f"[Server] lesson error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Chunk simplification (three distinct strategies via `hint`) ───────────────
@app.route("/api/simplify", methods=["POST"])
def simplify():
    chunk   = request.json.get("chunk", "")
    topic   = request.json.get("topic", "")
    hint    = request.json.get("hint", "analogy")
    attempt = request.json.get("attempt", 1)

    if hint == "analogy":
        instruction = (
            f"The student is confused by this concept from a lesson on '{topic}'. "
            f"Re-explain it using one concrete real-world analogy they would instantly recognise. "
            f"Do not repeat the original wording. Under 3 sentences, plain prose only."
        )
    elif hint == "eli5":
        instruction = (
            f"The student is still confused about this concept from '{topic}'. "
            f"Explain it as if they are 10 years old. Use the simplest possible language, "
            f"zero jargon, and one relatable everyday example. Under 3 sentences."
        )
    else:  # stepbystep
        instruction = (
            f"The student is really struggling with this concept from '{topic}'. "
            f"Break it down into a numbered sequence of 3 to 5 simple steps. "
            f"Each step is one plain sentence. No jargon at all."
        )

    try:
        result = featherless._chat(
            system="You are a patient tutor. Follow the instruction exactly.",
            user=f"{instruction}\n\nOriginal text:\n{chunk}",
            max_tokens=200,
        )
        return jsonify({"simplified": result.strip()})
    except Exception as e:
        print(f"[Server] simplify error: {e}")
        return jsonify({"error": str(e)}), 500


# ── TTS: streams edge-tts MP3 directly to the browser ────────────────────────
@app.route("/api/speak", methods=["POST"])
def speak():
    text  = request.json.get("text", "").strip()
    voice = "en-US-AvaMultilingualNeural"
    if not text:
        return Response(b"", mimetype="audio/mpeg")

    async def generate():
        buf = io.BytesIO()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return buf.read()

    try:
        audio = asyncio.run(generate())
        return Response(audio, mimetype="audio/mpeg")
    except Exception as e:
        print(f"[Server] TTS error: {e}")
        return Response(b"", mimetype="audio/mpeg")


# ── STT: records mic until silence, returns transcript ────────────────────────
@app.route("/api/listen", methods=["POST"])
def listen():
    try:
        text = stt.listen(prompt="educational question")
        if text:
            print(f"[Server] STT heard: \"{text}\"")
            return jsonify({"text": text})
        else:
            return jsonify({"error": "Nothing heard"}), 204
    except Exception as e:
        print(f"[Server] STT error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Image: asks Featherless for the best search query for a chunk ─────────────
@app.route("/api/generate_image", methods=["POST"])
def generate_image():
    chunk = request.json.get("chunk", "")
    topic = request.json.get("topic", "")

    try:
        query = featherless._chat(
            system=(
                "You are a visual learning assistant. "
                "Respond with only a short image search query, 5 words max, no punctuation."
            ),
            user=(
                f"Best image search query to visually explain this concept:\n\n"
                f"Topic: {topic}\nConcept: {chunk}"
            ),
            max_tokens=20,
        ).strip().strip('"')
        return jsonify({"query": query})
    except Exception as e:
        print(f"[Server] generate_image error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Video: triggers OpenNote video generation (30–120 s) ─────────────────────
@app.route("/api/generate_video", methods=["POST"])
def generate_video():
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    try:
        l = opennote.generate_lesson(topic, generate_video=True)
        print(f"[Server] video_url = {l.video_url}")
        if not l.video_url:
            return jsonify({"error": "OpenNote returned no video URL"}), 500
        return jsonify({"video_url": l.video_url})
    except Exception as e:
        print(f"[Server] generate_video error: {e}")
        return jsonify({"error": str(e)}), 500



# -- Camera monitor: start/stop from browser toggle ---------------------------
# Browser getUserMedia and OpenCV cannot share the camera simultaneously.
# We start the Python monitor only when the user clicks Start Camera,
# and stop it (releasing the device) when they stop the camera.

@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    if monitor.is_running():
        return jsonify({"ok": True, "msg": "already running"})
    ok = monitor.start()
    return jsonify({"ok": ok, "msg": "started" if ok else "failed to open camera"})


@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    if monitor.is_running():
        monitor.stop()
    _face_state.update({"state": "unknown", "confidence": 0.0, "note": ""})
    return jsonify({"ok": True})


# ── Face state: returns latest FacialMonitor reading ─────────────────────────
@app.route("/api/face_state")
def face_state():
    return jsonify(_face_state)


# ── Facial signal (from FacialMonitor → lesson_manager) ──────────────────────
@app.route("/api/signal", methods=["POST"])
def signal():
    sig = request.json.get("signal", "")
    manager.signal(sig)
    return jsonify({"ok": True})


# ── Lesson state ──────────────────────────────────────────────────────────────
@app.route("/api/state")
def state():
    return jsonify({
        "state": manager.state.name,
        "chunk": manager.session.current_chunk_index if manager.session else 0,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(port=5000, debug=False)