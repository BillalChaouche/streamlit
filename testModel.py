import streamlit as st
from transformers import pipeline
from pydub import AudioSegment
import tempfile
import os
import mysql.connector

# Initialize the pipeline for transcribing audio
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# MySQL connection setup
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="streamlit"
    )

def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcraped_video (
            id INT AUTO_INCREMENT PRIMARY KEY,
            video_path VARCHAR(255) NOT NULL,
            transcription TEXT NOT NULL
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Call create_table once at the beginning
create_table()

def convert_audio_to_wav(input_file):
    """Convert audio file to WAV format."""
    audio = AudioSegment.from_file(input_file)
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(output_file.name, format="wav")
    return output_file.name

def transcribe(audio_path):
    """Transcribe the audio file."""
    result = pipe(audio_path)
    return result["text"]

def save_transcription_to_db(video_path, transcription):
    """Save the video path and transcription to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transcraped_video (video_path, transcription)
        VALUES (%s, %s)
    """, (video_path, transcription))
    conn.commit()
    cursor.close()
    conn.close()

def test_page():
    st.title("Audio Transcription with Whisper")

    st.header("Upload an Audio File")
    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac", "m4a"])

    if audio_file is not None:
        st.audio(audio_file, format=audio_file.type)

        # Ensure user_videos directory exists
        if not os.path.exists("user_videos"):
            os.makedirs("user_videos")

        # Generate a unique file name
        file_counter = len(os.listdir("user_videos")) + 1
        unique_filename = f"user_videos/audio_{file_counter}.{audio_file.type.split('/')[1]}"

        # Save the uploaded audio file
        with open(unique_filename, "wb") as f:
            f.write(audio_file.read())

        # Convert the audio file to WAV format if necessary
        if audio_file.type != "audio/wav":
            wav_audio_path = convert_audio_to_wav(unique_filename)
        else:
            wav_audio_path = unique_filename

        # Transcribe the audio
        with st.spinner("Transcribing..."):
            transcription = transcribe(wav_audio_path)

        # Save the transcription to the database
        save_transcription_to_db(unique_filename, transcription)

        st.markdown("**Transcription:**")
        st.text_area("Transcribed Text", value=transcription, height=200)

if __name__ == "__main__":
    test_page()
