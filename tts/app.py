from flask import Flask, request, send_file
from transformers import AutoModel, AutoProcessor
import torch
import scipy.io.wavfile
import tempfile

app = Flask(__name__)
processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark").to("cuda")

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get('text')
    inputs = processor(text, return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key].to("cuda")
    audio_array = model.generate(**inputs).cpu().numpy().squeeze()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, 24000, audio_array)
        return send_file(tmpfile.name, mimetype='audio/wav')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
