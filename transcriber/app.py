import os
import tempfile
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

import whisperx
from whisperx.diarize import DiarizationPipeline

# Get API key, device, model, and Hugging Face token from environment variables
API_KEY = os.getenv("TRANSCRIBE_API_KEY", "1234")
DEVICE = "cuda"
MODEL = os.getenv("MODEL", "medium")
HF_TOKEN = os.getenv("HF_TOKEN")  # Your Hugging Face token for diarization

# Initialize the Flask app and allow cross-origin requests
app = Flask(__name__)
CORS(app)

# Initialize WhisperX ASR model (Automatic Speech Recognition)
asr_model = whisperx.load_model(MODEL, device=DEVICE, compute_type="float16")

# Initialize the speaker diarization pipeline
diarize_pipeline = DiarizationPipeline(
    use_auth_token=HF_TOKEN,
    device=DEVICE
)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    start_time = time.time()
    # Check the request API key
    key = request.headers.get('X-API-KEY')
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # Make sure the uploaded audio file is present in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']

    # Save the uploaded audio temporarily for processing
    with tempfile.NamedTemporaryFile(suffix=".tmp") as temp_audio:
        audio_file.save(temp_audio.name)

        # Step 1: ASR transcription
        audio = whisperx.load_audio(temp_audio.name)
        asr_result = asr_model.transcribe(audio, batch_size=16)

        # Step 2: (Optional) Word-level alignment for higher accuracy
        # Uncomment below if you need word-level time alignment:
        # align_model, metadata = whisperx.load_align_model(language_code=asr_result["language"], device=DEVICE)
        # asr_result = whisperx.align(asr_result["segments"], align_model, metadata, audio, DEVICE)

        # Step 3: Speaker diarization (detect who speaks when)
        diarization_segments = diarize_pipeline(audio)
        # Assign speaker labels to each transcription segment
        result = whisperx.assign_word_speakers(diarization_segments, asr_result)

        # Gather speaker, timestamps, and text for each segment
        segments = [
            {
                "speaker": seg.get('speaker', 'unknown'),
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }
            for seg in result["segments"]
        ]
    end_time = time.time()
    diff_seconds = round(end_time - start_time, 3)

    print(f"Exec time: {diff_seconds} seconds")
    # Return the transcription with diarization as JSON
    return jsonify({"segments": segments})

if __name__ == "__main__":
    # Start the Flask server, listening on all network interfaces on port 5005
    app.run(host="0.0.0.0", port=5005)
