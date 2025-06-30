# ===== ./app.py (PHIÊN BẢN HỢP NHẤT - SỬA LỖI API DECODING) =====
import os
import torch
import uuid
import json
import threading
import shutil
import requests
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from pydub import AudioSegment
from datetime import timedelta
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from asgiref.wsgi import WsgiToAsgi
from dotenv import load_dotenv

load_dotenv()

# --- KHỞI TẠO CÁC DỊCH VỤ ---
# 1. DỊCH VỤ LOCAL
try:
    print("🚀 Đang tải model XTTS-v2 cho chế độ LOCAL...")
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_local = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
    actual_tokenizer = tts_local.synthesizer.tts_model.tokenizer.tokenizer
    eos_token_id = actual_tokenizer.encode("[EOS]").ids[0]
    tts_local.synthesizer.tts_model.config.pad_token_id = eos_token_id
    print(f"✅ Model LOCAL đã sẵn sàng trên thiết bị: {device.upper()}!")
except Exception as e:
    print(f"⚠️ CẢNH BÁO: Không thể tải model LOCAL. Chế độ Local sẽ không khả dụng. Lỗi: {e}")
    tts_local = None

# 2. DỊCH VỤ API
try:
    print("🚀 Đang khởi tạo client cho chế độ API...")
    from tts_api_client import api_client, CoquiAPIError
except ImportError as e:
    print(f"⚠️ CẢNH BÁO: Không thể import tts_api_client. Chế độ API sẽ không khả dụng. Lỗi: {e}")
    class CoquiAPIError(Exception): pass
    class MockApiClient:
        def generate(self, *args, **kwargs): raise CoquiAPIError("Client API thật chưa được triển khai.")
    api_client = MockApiClient()

# --- CẤU HÌNH FLASK & HẰNG SỐ ---
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.urandom(24),
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
    UPLOAD_FOLDER='/tmp/uploads',
    OUTPUT_FOLDER='/tmp/outputs'
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

LANGUAGES_MASTER = { "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish", "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese", "ko": "Korean", "hu": "Hungarian", "hi": "Hindi" }
DEFAULT_VOICES = { "male_zh": {"name": "Chinese Male", "path": "samples/male_zh.mp3"}, "female_zh": {"name": "Chinese Female", "path": "samples/female_zh.mp3"} }
DEFAULT_ADVANCED_PARAMS = { 'temperature': 0.75, 'speed': 1.0, 'top_k': 50, 'top_p': 0.85, 'repetition_penalty': 5.0, 'length_penalty': 1.0 }
UI_STRINGS = {
    "en": { "script_example": "A: Hello world.\nB: Voice cloning is amazing!", "title": "XTTS Conversation Generator (Local + API)", "subtitle": "Choose Local mode for on-premise processing or API mode for cloud service.", "manage_speakers": "Manage Speakers", "add_speaker": "Add Speaker", "script_label": "Conversation Script", "script_placeholder": "Example:\nA: Hello world.\nB: Voice cloning is amazing!", "advanced_settings": "Advanced Settings", "generate_button": "Generate Voice", "clear_button_title": "Delete all session data and files", "results_header": "Conversation Results", "loading_generating": "Generating voice...", "loading_sending": "Sending request...", "loading_starting": "Starting...", "loading_processing": "Processing...", "error_checking_status": "Error checking status", "error_starting_job": "Invalid response from server when starting job", "error_occurred": "An error occurred", "confirm_clear": "Are you sure you want to delete all session data?", "error_clearing": "Error during cleanup", "full_conversation": "Full Conversation", "individual_lines": "Individual Lines", "no_results": "No results generated.", "speaker_id_prefix": "Speaker", "voice_select_placeholder": "-- Select Voice --", "voice_source_default": "Default", "voice_source_custom": "Custom", "current_file": "Current file", "no_file_uploaded": "No file uploaded yet.", "generation_mode": "Generation Mode", "local_mode": "Local", "api_mode": "API"},
    "vi": { "script_example": "A: Chào thế giới.\nB: Công nghệ nhân bản giọng nói thật tuyệt vời!", "title": "Trình tạo Hội thoại XTTS (Local + API)", "subtitle": "Chọn chế độ Local để xử lý tại chỗ hoặc API để dùng dịch vụ ngoài", "manage_speakers": "Quản lý Người nói", "add_speaker": "Thêm Người nói", "script_label": "Kịch bản Hội thoại", "script_placeholder": "Ví dụ:\nA: Chào thế giới.\nB: Công nghệ nhân bản giọng nói thật tuyệt vời!", "advanced_settings": "Cài đặt Nâng cao", "generate_button": "Tạo Giọng nói", "clear_button_title": "Xóa toàn bộ session và file", "results_header": "Kết quả Hội thoại", "loading_generating": "Đang tạo giọng nói...", "loading_sending": "Đang gửi yêu cầu...", "loading_starting": "Đang bắt đầu...", "loading_processing": "Đang xử lý...", "error_checking_status": "Lỗi khi kiểm tra trạng thái", "error_starting_job": "Phản hồi không hợp lệ từ máy chủ khi bắt đầu công việc", "error_occurred": "Đã xảy ra lỗi", "confirm_clear": "Bạn có chắc muốn xóa toàn bộ dữ liệu phiên làm việc?", "error_clearing": "Có lỗi xảy ra khi dọn dẹp", "full_conversation": "Toàn bộ Hội thoại", "individual_lines": "Từng câu thoại", "no_results": "Không có kết quả nào được tạo ra.", "speaker_id_prefix": "Người nói", "voice_select_placeholder": "-- Chọn giọng --", "voice_source_default": "Mặc định", "voice_source_custom": "Tùy chỉnh", "current_file": "File hiện tại", "no_file_uploaded": "Chưa có file nào được tải lên.", "generation_mode": "Chế độ Tạo", "local_mode": "Local", "api_mode": "API"},
    "zh": { "script_example": "A: 你好世界。\nB: 语音克隆技术真是太棒了！", "title": "XTTS 对话生成器 (本地+API)", "subtitle": "选择“本地”模式进行本地处理，或选择“API”模式使用云服务", "manage_speakers": "说话人管理", "add_speaker": "添加说话人", "script_label": "对话脚本", "script_placeholder": "示例:\nA: 你好世界。\nB: 语音克隆技术真是太棒了！", "advanced_settings": "高级参数设置", "generate_button": "生成语音", "clear_button_title": "删除所有会话数据和文件", "results_header": "对话结果", "loading_generating": "正在生成语音...", "loading_sending": "发送请求中...", "loading_starting": "开始中...", "loading_processing": "处理中...", "error_checking_status": "检查状态时出错", "error_starting_job": "从服务器收到的启动作业响应无效", "error_occurred": "发生错误", "confirm_clear": "您确定要删除所有会话数据吗？", "error_clearing": "清理会话时出错", "full_conversation": "完整对话", "individual_lines": "单句", "no_results": "没有生成结果。", "speaker_id_prefix": "说话人", "voice_select_placeholder": "-- 选择声音 --", "voice_source_default": "默认", "voice_source_custom": "自定义", "current_file": "当前文件", "no_file_uploaded": "尚未上传文件。", "generation_mode": "生成模式", "local_mode": "本地", "api_mode": "API"}
}
SUPPORTED_UI_LANGS = {"en": "English", "vi": "Tiếng Việt", "zh": "简体中文"}
jobs: Dict[str, Dict] = {}


# --- CÁC HÀM XỬ LÝ LÕI ---

def run_tts_job_local(job_id: str, job_data: dict):
    try:
        job_data['job_id'] = job_id
        lines = [l.strip() for l in job_data['script'].strip().split("\n") if l.strip() and ':' in l]
        if not lines: raise ValueError("Kịch bản trống hoặc không hợp lệ.")
        jobs[job_id]['status'] = 'processing'
        temp_results = []
        app_config = {'OUTPUT_FOLDER': app.config['OUTPUT_FOLDER'], 'UPLOAD_FOLDER': app.config['UPLOAD_FOLDER']}
        max_workers = 1 if device == "cpu" else 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(line, i, len(lines), job_data, app_config) for i, line in enumerate(lines)]
            futures = [executor.submit(process_single_line_local, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result: temp_results.append(result)
        temp_results.sort(key=lambda x: x['index'])
        audio_files_data = [r['data'] for r in temp_results]
        paths_to_combine = [r['disk_path'] for r in temp_results]
        jobs[job_id]['progress'] = f"Đang gộp {len(paths_to_combine)} file audio..."
        combined_path, combined_url = combine_audio_files(paths_to_combine)
        final_output_files = paths_to_combine
        if combined_path: final_output_files.append(combined_path)
        jobs[job_id]['generated_output_files'] = final_output_files
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {"results": audio_files_data, "conversation_audio": combined_url}
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = f"Lỗi (Local): {str(e)}"

def process_single_line_local(args):
    line, index, total_lines, job_data, app_config = args
    jobs[job_data['job_id']]['progress'] = f"Local: Dòng {index+1}/{total_lines} trên {device.upper()}..."
    speaker_id, text = [p.strip() for p in line.split(":", 1)]
    speaker_id = speaker_id.upper()
    lang_code = job_data.get('language_to_generate')
    adv_params = job_data.get('advanced_params', {})
    config = job_data['speakers_config'][speaker_id]
    speaker_path = None
    if config.get('voice_source') == 'uploaded':
        speaker_path = job_data['voice_map'].get(f"uploaded_{speaker_id}")
    else:
        speaker_path = job_data['voice_map'].get(config.get('voice_id'))
    if not speaker_path or not os.path.exists(speaker_path):
        raise ValueError(f"Không tìm thấy file giọng cho người nói '{speaker_id}'.")
    output_filename = f"{uuid.uuid4()}.wav"
    output_path_on_disk = os.path.join(app_config['OUTPUT_FOLDER'], output_filename)
    tts_local.tts_to_file(text=text, speaker_wav=speaker_path, language=lang_code, file_path=output_path_on_disk,
                          speed=float(adv_params.get('speed', 1.0)), temperature=float(adv_params.get('temperature', 0.75)),
                          top_p=float(adv_params.get('top_p', 0.85)), top_k=int(adv_params.get('top_k', 50)),
                          repetition_penalty=float(adv_params.get('repetition_penalty', 5.0)),
                          length_penalty=float(adv_params.get('length_penalty', 1.0)))
    return {"index": index, "data": {"speaker": speaker_id, "text": text, "path": f"/outputs/{output_filename}"}, "disk_path": output_path_on_disk}

def run_tts_job_api(job_id: str, job_data: dict):
    try:
        job_data['job_id'] = job_id
        lines = [l.strip() for l in job_data['script'].strip().split("\n") if l.strip() and ':' in l]
        if not lines: raise ValueError("Kịch bản trống hoặc không hợp lệ.")
        jobs[job_id]['status'] = 'processing'
        temp_results = []
        app_config = {'OUTPUT_FOLDER': app.config['OUTPUT_FOLDER'], 'UPLOAD_FOLDER': app.config['UPLOAD_FOLDER']}
        with ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [(line, i, len(lines), job_data, app_config) for i, line in enumerate(lines)]
            futures = [executor.submit(process_single_line_api, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result: temp_results.append(result)
        temp_results.sort(key=lambda x: x['index'])
        audio_files_data = [r['data'] for r in temp_results]
        paths_to_combine = [r['disk_path'] for r in temp_results]
        jobs[job_id]['progress'] = f"Đang gộp {len(paths_to_combine)} file audio..."
        combined_path, combined_url = combine_audio_files(paths_to_combine)
        final_output_files = paths_to_combine
        if combined_path: final_output_files.append(combined_path)
        jobs[job_id]['generated_output_files'] = final_output_files
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {"results": audio_files_data, "conversation_audio": combined_url}
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = f"Lỗi (API): {str(e)}"

# ⭐⭐⭐ HÀM ĐÃ ĐƯỢC SỬA LỖI ⭐⭐⭐
def process_single_line_api(args):
    line, index, total_lines, job_data, app_config = args
    jobs[job_data['job_id']]['progress'] = f"API: Dòng {index+1}/{total_lines} (Chuẩn hóa & Upload)..."
    speaker_id, text = [p.strip() for p in line.split(":", 1)]
    speaker_id = speaker_id.upper()
    lang_code = job_data.get('language_to_generate')
    lang_name = LANGUAGES_MASTER.get(lang_code, "English")
    adv_params = job_data.get('advanced_params', {})
    config = job_data['speakers_config'][speaker_id]
    speaker_path = None
    if config.get('voice_source') == 'uploaded':
        speaker_path = job_data['voice_map'].get(f"uploaded_{speaker_id}")
    else:
        speaker_path = job_data['voice_map'].get(config.get('voice_id'))
    if not speaker_path or not os.path.exists(speaker_path):
        raise ValueError(f"Không tìm thấy file giọng cho người nói '{speaker_id}'.")
    
    temp_ref_wav = None
    try:
        # Chuyển đổi file âm thanh sang định dạng WAV chuẩn trước khi upload
        temp_ref_wav = os.path.join(app_config['UPLOAD_FOLDER'], f"ref_api_{uuid.uuid4()}.wav")
        audio = AudioSegment.from_file(speaker_path)
        audio = audio.set_frame_rate(22050).set_channels(1)
        audio.export(temp_ref_wav, format="wav")

        # Upload file WAV đã được chuẩn hóa
        with open(temp_ref_wav, 'rb') as f:
            response = requests.post('https://tmpfiles.org/api/v1/upload', files={'file': f})
        response.raise_for_status()
        public_url = response.json()['data']['url'].replace('tmpfiles.org/', 'tmpfiles.org/dl/')

        # Gọi API với URL của file WAV
        standardized_result_path = api_client.generate(text=text, lang=lang_name, reference_wav_url=public_url, advanced_params=adv_params)
        
        output_filename = os.path.basename(standardized_result_path)
        output_path_on_disk = os.path.join(app_config['OUTPUT_FOLDER'], output_filename)
        shutil.move(standardized_result_path, output_path_on_disk)
        
        return {"index": index, "data": {"speaker": speaker_id, "text": text, "path": f"/outputs/{output_filename}"}, "disk_path": output_path_on_disk}
    finally:
        # Dọn dẹp file WAV tạm sau khi hoàn tất
        if temp_ref_wav and os.path.exists(temp_ref_wav):
            os.remove(temp_ref_wav)

# --- CÁC ROUTE CỦA FLASK APP ---
@app.route("/")
def index_route():
    new_ui_lang = request.args.get('lang')
    current_ui_lang = session.get('ui_lang', 'zh')
    if new_ui_lang and new_ui_lang in SUPPORTED_UI_LANGS and new_ui_lang != current_ui_lang:
        session['ui_lang'] = new_ui_lang
        session.pop('script', None)
    final_ui_lang = session.get('ui_lang', 'zh')
    if final_ui_lang not in UI_STRINGS: final_ui_lang = 'zh'
    strings = UI_STRINGS[final_ui_lang]
    current_params = {**DEFAULT_ADVANCED_PARAMS, **session.get('advanced_params', {})}
    session_state = {
        'speakers': session.get('speakers', {'A': {'voice_source': 'default', 'voice_id': 'male_zh'}, 'B': {'voice_source': 'default', 'voice_id': 'female_zh'}}),
        'uploaded_voices': session.get('uploaded_voices', {}),
        'script': session.get('script', strings['script_example']),
        'language_to_generate': session.get('language_to_generate', 'zh-cn'),
        'advanced_params': current_params,
        'generation_mode': session.get('generation_mode', 'local') 
    }
    session.permanent = True
    return render_template("index.html", ui_lang=final_ui_lang, ui_strings=strings, supported_ui_langs=SUPPORTED_UI_LANGS, 
                           languages_master=LANGUAGES_MASTER, default_voices=DEFAULT_VOICES, session_state=session_state)

@app.route("/generate", methods=["POST"])
def generate():
    clear_old_output_files()
    data = request.form
    mode = data.get('generation_mode', 'local')
    session['generation_mode'] = mode
    if mode == 'local' and not tts_local:
        return jsonify({"error": "Chế độ Local không khả dụng. Model chưa được tải."}), 503
    if mode == 'api' and not (hasattr(api_client, 'client') and api_client.client):
        return jsonify({"error": "Chế độ API không khả dụng. Không thể kết nối tới dịch vụ."}), 503
    session['script'] = data.get('script')
    session['speakers'] = json.loads(data.get('speakers_config', '{}'))
    session['language_to_generate'] = data.get('language_to_generate')
    advanced_params = {}
    for key in DEFAULT_ADVANCED_PARAMS.keys():
        try: advanced_params[key] = float(data.get(key))
        except (ValueError, TypeError): advanced_params[key] = DEFAULT_ADVANCED_PARAMS[key]
    session['advanced_params'] = advanced_params
    session.setdefault('uploaded_voices', {})
    voice_map = {key: value['path'] for key, value in DEFAULT_VOICES.items()}
    for sid, vdata in session.get('uploaded_voices', {}).items():
        voice_map[f"uploaded_{sid}"] = vdata['path']
    for fkey, fstorage in request.files.items():
        if fstorage.filename != '':
            sid = fkey.split('_')[-1]
            if sid in session['uploaded_voices'] and os.path.exists(session['uploaded_voices'][sid]['path']):
                try: os.remove(session['uploaded_voices'][sid]['path'])
                except OSError: pass
            save_path, _ = handle_file_upload(fstorage)
            session['uploaded_voices'][sid] = {'path': save_path, 'name': os.path.basename(fstorage.filename)}
            voice_map[f"uploaded_{sid}"] = save_path
    session.modified = True
    job_id = str(uuid.uuid4())
    job_data = {'script': session['script'], 'language_to_generate': session['language_to_generate'],
                'speakers_config': session['speakers'], 'voice_map': voice_map, 'advanced_params': session['advanced_params']}
    jobs[job_id] = {'status': 'queued', 'progress': 'Đang chờ xử lý...'}
    target_func = run_tts_job_local if mode == 'local' else run_tts_job_api
    thread = threading.Thread(target=target_func, args=(job_id, job_data))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "processing", "job_id": job_id, 
                    "updated_session_state": session.to_dict() if hasattr(session, 'to_dict') else dict(session)})

@app.route("/status/<job_id>")
def get_status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({"status": "failed", "error": "Job ID không tồn tại."}), 404
    if job.get('status') in ['completed', 'failed']:
        job_to_return = job.copy()
        session.setdefault('files_to_cleanup_on_next_run', []).extend(job.get('generated_output_files', []))
        session.modified = True
        jobs.pop(job_id, None)
        return jsonify(job_to_return)
    return jsonify(job)

@app.route("/clear_all")
def clear_all_session_data():
    try:
        for folder in [app.config['OUTPUT_FOLDER'], app.config['UPLOAD_FOLDER']]:
            if os.path.exists(folder):
                shutil.rmtree(folder); os.makedirs(folder, exist_ok=True)
        session.clear(); global jobs; jobs.clear()
        return jsonify({"status": "success", "message": "Toàn bộ session và file đã được xóa."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Có lỗi xảy ra khi dọn dẹp: {e}"}), 500

def combine_audio_files(file_paths):
    if not file_paths: return None, None
    combined = AudioSegment.empty()
    for path in file_paths:
        if os.path.exists(path):
            try: combined += AudioSegment.from_file(path)
            except Exception as e: print(f"Lỗi khi đọc file {path}: {e}"); continue
    if len(combined) == 0: return None, None
    filename = f"conversation_{uuid.uuid4()}.wav"
    disk_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    combined.export(disk_path, format="wav", parameters=["-ar", "24000"])
    return disk_path, f"/outputs/{filename}"

def clear_old_output_files():
    files_to_delete = session.pop('files_to_cleanup_on_next_run', [])
    for file_path in files_to_delete:
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except OSError as e: print(f"Lỗi khi xóa file output {file_path}: {e}")

def handle_file_upload(file):
    filename = f"upload_{uuid.uuid4()}_{os.path.basename(file.filename)}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return save_path, os.path.basename(file.filename)

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# --- KHỞI CHẠY APP ---
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:asgi_app", host="0.0.0.0", port=7861, reload=True)