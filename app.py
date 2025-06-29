# ===== ./app.py (PHIÊN BẢN HOÀN THIỆN) =====
from flask import Flask, render_template, request, jsonify, session
import os
import torch
from TTS.api import TTS
import uuid
from pydub import AudioSegment
from datetime import timedelta
import json

# ... (Các phần import an toàn giữ nguyên) ...
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


print("🚀 Đang tải model XTTS v2... Vui lòng chờ.")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
print(f"✅ Model đã sẵn sàng trên thiết bị: {device.upper()}!")

DEFAULT_VOICES = {
    "male_zh_mp3": {"name": "Giọng Nam (Tiếng Trung)", "path": "samples/male_zh.mp3"},
    "female_zh_mp3": {"name": "Giọng Nữ (Tiếng Trung)", "path": "samples/female_zh.mp3"},
}
SUPPORTED_LANGUAGES = { "zh-cn": "Chinese (Simplified)", "en": "English", "vi": "Vietnamese" }

def clear_output_files():
    """Xóa các file audio ĐẦU RA (outputs) được tạo trong phiên trước đó."""
    if 'generated_output_files' in session:
        for file_path in session.get('generated_output_files', []):
            try:
                if os.path.exists(file_path): os.remove(file_path)
            except Exception as e: print(f"Lỗi khi xóa file output {file_path}: {e}")
    session['generated_output_files'] = []

@app.route("/")
def index():
    session_state = {
        'speakers': session.get('speakers', {
            'A': {'voice_source': 'default', 'voice_id': 'male_zh_mp3'},
            'B': {'voice_source': 'default', 'voice_id': 'female_zh_mp3'}
        }),
        'uploaded_voices': session.get('uploaded_voices', {}),
        'script': session.get('script', 'A: 你好。\nB: 你好吗？'),
        'language': session.get('language', 'zh-cn'),
        'speed': session.get('speed', 1.0)
    }
    session.permanent = True
    return render_template("index.html", 
                           languages=SUPPORTED_LANGUAGES, 
                           default_voices=DEFAULT_VOICES, 
                           session_state=session_state)

@app.route("/generate", methods=["POST"])
def generate():
    clear_output_files()
    
    data = request.form
    script = data.get('script')
    language = data.get('language')
    speed = data.get('speed', 1.0, type=float)
    speakers_config = json.loads(data.get('speakers_config', '{}'))

    session['script'] = script
    session['language'] = language
    session['speed'] = speed
    session['speakers'] = speakers_config
    session.setdefault('uploaded_voices', {})

    voice_map = {key: value['path'] for key, value in DEFAULT_VOICES.items()}
    for speaker_id, voice_data in session['uploaded_voices'].items():
        voice_map[f"uploaded_{speaker_id}"] = voice_data['path']

    for file_key, file_storage in request.files.items():
        if file_storage.filename != '':
            speaker_id = file_key.split('_')[-1]
            if speaker_id in session['uploaded_voices']:
                old_path = session['uploaded_voices'][speaker_id]['path']
                try:
                    if os.path.exists(old_path): os.remove(old_path)
                except Exception as e:
                    print(f"Lỗi khi xóa file cũ {old_path}: {e}")
            save_path, original_name = handle_file_upload(file_storage)
            session['uploaded_voices'][speaker_id] = {'path': save_path, 'name': original_name}
            voice_map[f"uploaded_{speaker_id}"] = save_path

    lines = [l.strip() for l in script.strip().split("\n") if l.strip()]
    audio_files_data = []
    newly_created_paths = []

    for line in lines:
        parts = line.split(":", 1)
        if len(parts) != 2: continue
        
        speaker_id = parts[0].strip().upper()
        text = parts[1].strip()

        if speaker_id not in speakers_config:
            return jsonify({"error": f"Người nói '{speaker_id}' trong kịch bản chưa được định nghĩa trong phần Quản lý Người nói."}), 400
        
        config = speakers_config[speaker_id]
        speaker_path = None
        
        if config.get('voice_source') == 'uploaded':
            speaker_path = voice_map.get(f"uploaded_{speaker_id}")
        elif config.get('voice_source') == 'default':
            speaker_path = voice_map.get(config.get('voice_id'))

        if not speaker_path or not os.path.exists(speaker_path):
            print(f"Lỗi: Không tìm thấy file giọng nói cho người nói '{speaker_id}'. Cấu hình nhận được: {config}")
            return jsonify({
                "error": f"Chưa chọn giọng nói hợp lệ cho người nói '{speaker_id}'. Vui lòng chọn một giọng trong danh sách hoặc tải lên file âm thanh."
            }), 400
            
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        try:
            tts.tts_to_file(text=text, speaker_wav=speaker_path, language=language, file_path=output_path, speed=speed)
            session['generated_output_files'].append(output_path)
            newly_created_paths.append(output_path)
            audio_files_data.append({"speaker": speaker_id, "text": text, "path": output_path})
        except Exception as e:
            print(f"Lỗi khi tạo TTS: {e}")
            return jsonify({"error": f"Không thể xử lý dòng: '{line}'. Chi tiết: {str(e)}"}), 500
            
    combined_path = combine_audio_files(newly_created_paths)
    if combined_path:
        session['generated_output_files'].append(combined_path)

    session.modified = True

    return jsonify({
        "results": audio_files_data,
        "conversation_audio": combined_path,
        "updated_session_state": session.to_dict() if hasattr(session, 'to_dict') else dict(session)
    })

@app.route("/clear_all")
def clear_all_session_data():
    clear_output_files()
    if 'uploaded_voices' in session:
        for voice_data in session.get('uploaded_voices', {}).values():
            try:
                if os.path.exists(voice_data['path']): os.remove(voice_data['path'])
            except Exception as e: print(f"Lỗi khi xóa file upload {voice_data['path']}: {e}")
    session.clear()
    return jsonify({"status": "success", "message": "Toàn bộ session và file đã được xóa."})

def handle_file_upload(file):
    filename = f"upload_{uuid.uuid4()}_{os.path.basename(file.filename)}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return save_path, os.path.basename(file.filename)

def combine_audio_files(file_paths):
    if not file_paths: return None
    combined = AudioSegment.empty()
    for path in file_paths:
        try: combined += AudioSegment.from_wav(path)
        except Exception as e: print(f"Lỗi khi đọc file {path} để nối: {e}"); continue
    combined_filename = f"conversation_{uuid.uuid4()}.wav"
    combined_path = os.path.join(app.config['OUTPUT_FOLDER'], combined_filename)
    combined.export(combined_path, format="wav")
    return combined_path

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)