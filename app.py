import os
import hashlib
import soundfile as sf
from flask import Flask, request, render_template
from TTS.api import TTS

app = Flask(__name__)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)

SPEAKER_MALE = "speakers/male.wav"
SPEAKER_FEMALE = "speakers/female.wav"

os.makedirs("static/audio", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    audio_files = []
    script_input = ""
    if request.method == "POST":
        script_input = request.form["script"]
        lines = script_input.strip().split("\n")
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            if line.startswith("A:"):
                speaker_wav = SPEAKER_FEMALE
            elif line.startswith("B:"):
                speaker_wav = SPEAKER_MALE
            else:
                continue
            text = line.split(":", 1)[1].strip()
            filename = hashlib.md5((text + speaker_wav).encode()).hexdigest() + ".wav"
            filepath = os.path.join("static/audio", filename)
            if not os.path.exists(filepath):
                wav = tts.tts(text, speaker_wav=speaker_wav, language="zh")
                sf.write(filepath, wav, 24000)
            audio_files.append({"path": filepath, "text": text})
    return render_template("index.html", audio_files=audio_files, script=script_input)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
