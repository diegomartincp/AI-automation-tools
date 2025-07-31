# WhisperX Diarization API (Dockerized)

A production-ready REST API for **automatic speech recognition (ASR)** and **speaker diarization** using WhisperX and Pyannote, packaged in Docker for convenient deployment and GPU acceleration.

## Features

- 🎤 **Transcribes audio** files (mp3, wav, opus, etc.) to text, leveraging WhisperX for fast and accurate ASR.
- 🗣️ **Speaker diarization**: Segments the transcription and labels each part according to the speaker detected (e.g., "SPEAKER_0", "SPEAKER_1", ...).
- 🐳 **Fully containerized**: Runs in an isolated Docker container, CUDA-ready for GPU inference.
- 🔐 **API Key protection**: Endpoint protected by an API key passed via header.
- ⚙️ **Customizable:** Configure Whisper model (tiny, base, medium, etc.), and Hugging Face token for diarization via environment variables or `.env` file.
- 🌍 **CORS support**: Ready for integration with n8n or other automation/orchestration tools.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-user/your-repository.git
```

### 2. Prepare your environment variables

Modify the docker compose file with your API key and Hugging Face token.
```bash
nano docker-compose.yml
```
- Get your Hugging Face token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- To use diarization, also visit and accept the model terms at [https://hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) using your Hugging Face account.

### 3. Build and run with Docker Compose
```bash
docker compose up --build
```
The API will be available at `http://localhost:5005/transcribe`

---

## API Usage

### Endpoint

`POST /transcribe`

- **Headers:**  
  `X-API-KEY: your_secret_api_key`
- **Body:**  
  Multipart form with `audio` field (the audio file to transcribe)

#### Example with `curl`:

```curl
curl -X POST http://localhost:5005/transcribe
-H "X-API-KEY: your_secret_api_key"
-F "audio=@/path/to/yourfile.mp3"
```
