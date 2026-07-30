from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from faster import predict, extract_text_from_audio, extract_text_from_video

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mov'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Prediction route (text + file upload)
@app.route('/predict', methods=['POST'])
def predict_text():
    try:
        # Handle text input
        input_text = request.form.get('text')
        if input_text:
            label, confidence = predict(input_text)
            return jsonify({'label': label, 'confidence': f"{confidence:.2f}"})

        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(tempfile.gettempdir(), filename)
                file.save(file_path)

                print(f"[DEBUG] File saved successfully: {file_path}")

                # Check file type and process accordingly
                file_ext = filename.rsplit('.', 1)[1].lower()
                text = None
                if file_ext in ['wav', 'mp3']:
                    print("[DEBUG] Processing audio file...")
                    text = extract_text_from_audio(file_path)
                elif file_ext in ['mp4', 'avi', 'mov']:
                    print("[DEBUG] Processing video file...")
                    text = extract_text_from_video(file_path)
                else:
                    return jsonify({'error': 'Unsupported file type'})

                os.unlink(file_path)  # Delete after processing

                if text and text not in ["Audio Processing Failed", "Video Processing Failed"]:
                    label, confidence = predict(text)
                    return jsonify({'label': label, 'confidence': f"{confidence:.2f}"})
                else:
                    return jsonify({'error': 'No speech could be transcribed from this file. Check that it contains clear speech and that FFmpeg and your internet connection are available.'})

        return jsonify({'error': 'No valid input provided'})
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return jsonify({'error': 'Error occurred. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)
