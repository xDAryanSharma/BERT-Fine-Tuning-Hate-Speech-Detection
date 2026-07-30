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

# Function to transcribe a WAV file with Google's speech-recognition service.
def _transcribe_wav(wav_path):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        return text.strip() or None
    except sr.UnknownValueError:
        print("[ERROR] Speech was not recognised in the uploaded media.")
        return None
    except sr.RequestError as error:
        print(f"[ERROR] Speech-recognition service request failed: {error}")
        return None
    except Exception as e:
        print(f"[ERROR] Speech transcription failed: {e}")
        return None


# Function to extract text from audio files.
def extract_text_from_audio(file_path):
    wav_path = None
    try:
        print(f"[DEBUG] Processing audio file: {file_path}")
        if not os.path.exists(file_path):
            print("[ERROR] File does not exist!")
            return "Audio Processing Failed"

        # SpeechRecognition reads WAV/AIFF/FLAC, so convert every accepted audio
        # type to a standard, mono WAV before transcription.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = temp_file.name
        ffmpeg.input(file_path).output(
            wav_path,
            format='wav',
            acodec='pcm_s16le',
            ac=1,
            ar=16000,
        ).run(overwrite_output=True, quiet=True)

        text = _transcribe_wav(wav_path)
        if not text:
            return "Audio Processing Failed"

        print(f"[DEBUG] Extracted Text from Audio: {text}")
        return text
    except ffmpeg.Error as error:
        details = error.stderr.decode(errors='replace') if error.stderr else str(error)
        print(f"[ERROR] Audio conversion failed: {details}")
        return "Audio Processing Failed"
    except Exception as error:
        print(f"[ERROR] Audio processing failed: {error}")
        return "Audio Processing Failed"
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

# Function to extract text from video files (Added Debugging)
def extract_text_from_video(file_path):
    temp_audio_path = None
    try:
        print(f"[DEBUG] Extracting audio from video: {file_path}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_audio_path = temp_file.name

        # Using ffmpeg to extract audio
        ffmpeg.input(file_path).output(temp_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar=16000).run(overwrite_output=True, quiet=True)

        print(f"[DEBUG] Extracted audio file: {temp_audio_path}")

        # Process extracted audio for speech recognition
        text = _transcribe_wav(temp_audio_path)

        return text if text else "Video Processing Failed"
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        return "Video Processing Failed"
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
