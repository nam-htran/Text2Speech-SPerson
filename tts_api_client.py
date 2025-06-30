# ===== ./tts_api_client.py =====
# Giữ nguyên nội dung file này từ phiên bản API trước đó.
# (File này đã được cung cấp trong câu hỏi trước, không cần dán lại ở đây)
import time
import uuid
import os
import shutil
from typing import Optional, Dict

try:
    from gradio_client import Client
except ImportError as e:
    raise ImportError(f"Không thể import gradio_client. Hãy đảm bảo bạn đã chạy 'pip install gradio_client'. Lỗi gốc: {e}")

try:
    from pydub import AudioSegment
except ImportError:
    raise ImportError("Không thể import pydub. Hãy đảm bảo bạn đã chạy 'pip install pydub'.")

class CoquiAPIError(Exception):
    """Lỗi tùy chỉnh cho các vấn đề liên quan đến API call."""
    pass

class CoquiAPIClient:
    def __init__(self, space_url="hasanbasbunar/Voice-Cloning-XTTS-v2"):
        print(f"Đang kết nối tới Gradio client: {space_url}...")
        try:
            self.client = Client(src=space_url)
            print("✅ Kết nối Gradio client thành công!")
        except Exception as e:
            print(f"⚠️ CẢNH BÁO: Không thể khởi tạo Gradio client: {e}")
            self.client = None

    def _standardize_output_to_wav(self, temp_api_path: str) -> str:
        try:
            audio = AudioSegment.from_file(temp_api_path)
            standardized_wav_path = os.path.join("/tmp", f"api_output_std_{uuid.uuid4()}.wav")
            audio.export(standardized_wav_path, format="wav")
            return standardized_wav_path
        except Exception as e:
            raise CoquiAPIError(f"Lỗi khi chuẩn hóa file âm thanh đầu ra: {e}")

    def generate(
        self, 
        text: str, 
        lang: str, 
        reference_wav_url: str,
        advanced_params: Optional[dict] = None
    ) -> str:
        if not self.client:
            raise CoquiAPIError("Gradio client chưa được khởi tạo thành công.")

        api_params = {
            "text": text,
            "reference_audio_url": reference_wav_url,
            "example_audio_name": None,
            "language": lang,
            "api_name": "/voice_clone_synthesis"
        }
        
        if advanced_params: api_params.update(advanced_params)

        print(f"Đang gửi yêu cầu tới API '{self.client.space_id}' với URL tham chiếu: {reference_wav_url}")
        try:
            temp_api_result_path = self.client.predict(**api_params)
            if not temp_api_result_path or not os.path.exists(temp_api_result_path):
                 raise CoquiAPIError(f"API không trả về đường dẫn file audio hợp lệ. Kết quả: {temp_api_result_path}")
            final_wav_path = self._standardize_output_to_wav(temp_api_result_path)
            return final_wav_path
        except Exception as e:
            raise CoquiAPIError(f"Lỗi khi thực hiện predict từ Gradio API: {e}")

# Khởi tạo client khi import
api_client = CoquiAPIClient()