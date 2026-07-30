import numpy as np
import tensorflow as tf
import os
import tempfile
import speech_recognition as sr
from transformers import TFBertForSequenceClassification, BertTokenizer
from deep_translator import GoogleTranslator
import ffmpeg  

# Load BERT model and tokenizer
try:
    model = TFBertForSequenceClassification.from_pretrained('saved_model')
    tokenizer = BertTokenizer.from_pretrained('saved_model')
except Exception as e:
    print(f"Model loading error: {e}")

# Function to preprocess input text (translation + tokenization)
def preprocess(sample_text):
    try:
        if not sample_text:
            raise ValueError("Empty input received")

        # Translate text to English
        translated_text = GoogleTranslator(source='auto', target='en').translate(sample_text)
        print(f"[DEBUG] Translated text: {translated_text}")

        # Tokenize using BERT tokenizer
        inputs = tokenizer(translated_text, return_tensors="tf", padding=True, truncation=True, max_length=512)
        return inputs
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None

# Function to predict using the BERT model
def predict(sample_text):
    try:
        inputs = preprocess(sample_text)
        if inputs is not None:
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = tf.nn.softmax(logits, axis=1).numpy()[0]
            label = np.argmax(prediction)
            confidence = prediction[label]
            labels = ["Hate Speech", "Offensive", "No Hate"]
            print(f"[DEBUG] Prediction: {labels[label]}, Confidence: {confidence:.2f}")
            return labels[label], confidence
        else:
            return "Error in processing", 0.0
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "Error occurred", 0.0

# Function to extract text from audio files (Added Debugging)
def extract_text_from_audio(file_path):
    try:
        print(f"[DEBUG] Processing audio file: {file_path}")

        # Check if file exists before processing
        if not os.path.exists(file_path):
            print("[ERROR] File does not exist!")
            return "Audio Processing Failed"

        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        if not text.strip():
            print("[ERROR] No text extracted from audio.")
            return "Audio Processing Failed"

        print(f"[DEBUG] Extracted Text from Audio: {text}")
        return text
    except Exception as e:
        print(f"[ERROR] Audio processing failed: {e}")
        return "Audio Processing Failed"

# Function to extract text from video files (Added Debugging)
def extract_text_from_video(file_path):
    try:
        print(f"[DEBUG] Extracting audio from video: {file_path}")
        
        temp_audio_path = file_path.rsplit(".", 1)[0] + ".wav"  # Replace extension with .wav

        # Using ffmpeg to extract audio
        ffmpeg.input(file_path).output(temp_audio_path, format='wav', acodec='pcm_s16le').run(overwrite_output=True)

        print(f"[DEBUG] Extracted audio file: {temp_audio_path}")

        # Process extracted audio for speech recognition
        text = extract_text_from_audio(temp_audio_path)

        # Clean up extracted audio file
        os.remove(temp_audio_path)

        return text if text else "Audio Extraction Failed"
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        return "Video Processing Failed"