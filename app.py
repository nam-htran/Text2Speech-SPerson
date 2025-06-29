from flask import Flask, render_template, request
import os
import torch
from TTS.api import TTS

from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# ✅ Khai báo tất cả class cần thiết để PyTorch cho phép unpickle
add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig
])
app = Flask(__name__)

# Model XTTS multilingual
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)

@app.route("/", methods=["GET", "POST"])
def index():
    audio_files = []
    if request.method == "POST":
        script = request.form["script"]
        lines = [l.strip() for l in script.strip().split("\n") if l.strip()]
        for i, line in enumerate(lines):
            if line.startswith("A:"):
                speaker_wav = "samples/male.wav"
                text = line.replace("A:", "").strip()
            elif line.startswith("B:"):
                speaker_wav = "samples/female.wav"
                text = line.replace("B:", "").strip()
            else:
                continue

            out_path = f"static/output_{i}.wav"
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language="zh",
                file_path=out_path
            )
            audio_files.append({ "text": line, "path": out_path })

        return render_template("index.html", audio_files=audio_files, script=script)

    return render_template("index.html", script="")

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(host="0.0.0.0", port=7860)
