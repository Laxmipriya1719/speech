import torch
import pyaudio
import numpy as np
import librosa
from model.vesper_model import VesperFinetuneWrapper  # Your model class
from utils import extract_features  # Assuming you have a function for feature extraction

# Load the trained model
model = VesperFinetuneWrapper()  # Initialize your model
model.load_state_dict(torch.load('emotion_model.h5'))  # Load your saved model weights
model.eval()  # Set model to evaluation mode

# Setup microphone stream
p = pyaudio.PyAudio()
rate = 16000  # Sample rate, adjust if necessary
channels = 1  # Mono audio
chunk_size = 1024  # Size of each audio chunk

# Start streaming from the microphone
stream = p.open(format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk_size)

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        # Record data from microphone
        data = stream.read(chunk_size)
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Extract features from the audio data (MFCC or others)
        features = extract_features(audio_data, rate)  # Adjust extract_features function as per your requirements
        
        # Add batch dimension (required for the model)
        features = np.expand_dims(features, axis=0)
        
        # Convert features to tensor
        input_tensor = torch.tensor(features, dtype=torch.float32)

        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)  # Assuming model outputs logits
            predicted_class = torch.argmax(output, dim=1).item()

        # Map predicted class to emotion label
        emotions = ["neutral", "happy", "angry", "sad", "fearful", "disgust", "surprised"]  # Adjust based on your model
        predicted_emotion = emotions[predicted_class]
        
        # Print the predicted emotion
        print(f"Predicted Emotion: {predicted_emotion}")

except KeyboardInterrupt:
    print("\nRecording stopped.")

finally:
    # Close the microphone stream
    stream.stop_stream()
    stream.close()
    p.terminate()
