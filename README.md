# XTTS Conversation Generator (Local + API)

A powerful web-based tool for generating multi-speaker audio conversations using the Coqui XTTS-v2 model. This application offers a flexible dual-mode operation: **Local Mode** for on-premise processing utilizing your own hardware (CPU or GPU), and **API Mode** which offloads the generation to a remote Gradio-based API service.

This tool is perfect for content creators, developers, and researchers who need to quickly prototype dialogues, create audio for videos, or experiment with voice cloning technology.

## ✨ Features

- **Dual Generation Modes**:
  - **🖥️ Local Mode**: Utilizes a locally loaded XTTS-v2 model for full control, privacy, and no API costs. Significantly accelerated on NVIDIA GPUs (CUDA).
  - **☁️ API Mode**: Connects to a remote Gradio API, ideal for users without powerful local hardware.
- **🗣️ Multi-Speaker Dialogues**: Craft conversations with multiple unique speakers in a single script (e.g., `A: Hello.`, `B: Hi there!`).
- **🎤 Advanced Voice Cloning**: Use default pre-set voices or upload your own audio file (at least 6 seconds) for any speaker to clone their voice.
- **🎛️ Fine-Tuning Controls**: Adjust advanced parameters like `temperature`, `speed`, `top_k`, `top_p`, and penalties to fine-tune the audio output.
- **🌐 Multilingual Support**:
  - **UI**: Supports English, Vietnamese, and Chinese.
  - **Generation**: Supports all languages available in the XTTS-v2 model.
- ** asynchronous Job Processing**: The backend processes generation requests in the background, allowing for a non-blocking user experience with real-time progress updates.
- **📦 Dev Container Included**: Comes with a pre-configured VS Code Development Container for a consistent and easy setup.

## 🌍 Live Demo

A live version of this application is deployed on Hugging Face Spaces. You can try it out directly in your browser:

> **[https://huggingface.co/spaces/namthse182380/TTS](https://huggingface.co/spaces/namthse182380/TTS)**

Please note that the public demo runs on shared hardware and may have a queue. For the best performance and privacy, running the application locally is recommended.

## ⚙️ Architecture Overview

The application is built with a Flask backend and a vanilla JavaScript frontend styled with Tailwind CSS.

1.  **Frontend (UI)**: The user interacts with the web interface to select a mode, configure speakers, input a script, and adjust parameters.
2.  **Flask Backend**:
    - Receives the generation request.
    - Creates a unique job ID and starts a background thread to handle the processing.
    - **In Local Mode**: The thread uses the `TTS` library to synthesize each line of the script using the local XTTS-v2 model on the CPU or GPU.
    - **In API Mode**: The thread normalizes the reference audio, uploads it to a temporary file service (`tmpfiles.org`) to get a public URL, and then calls the remote Gradio API for synthesis.
    - **Audio Combination**: After all lines are generated, `pydub` is used to combine them into a single conversation audio file.
3.  **Job Polling**: The frontend periodically polls a `/status/<job_id>` endpoint to get progress updates and retrieve the final audio files once the job is complete.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- `ffmpeg`: Required for audio processing with `pydub`.
  - **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install ffmpeg`
  - **macOS (Homebrew)**: `brew install ffmpeg`
  - **Windows**: Download from the official site and add to your system's PATH.
- **(Optional but Highly Recommended for Local Mode)**: An NVIDIA GPU with CUDA installed for significantly faster generation.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nam-htran/Text2Speech-SPerson
    cd Text2Speech-SPerson
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run the application in Local Mode, the XTTS-v2 model (several gigabytes) will be downloaded automatically. This may take some time.*

### Running the Application

Once the installation is complete, start the web server using Uvicorn:

```bash
uvicorn app:asgi_app --host 0.0.0.0 --port 7861 --reload
