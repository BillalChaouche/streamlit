import streamlit as st
from pytube import YouTube
import os
from moviepy.editor import VideoFileClip, AudioFileClip
import pymysql
import hashlib

# MySQL connection setup
def connect_to_mysql():
    host = "localhost"
    username = "root"
    password = "root"
    database = "streamlit"
    connection = pymysql.connect(host=host, user=username, password=password, database=database)
    return connection

# Authenticate the user
def authenticate(username, password):
    connection = connect_to_mysql()
    try:
        hashed_password = hashlib.md5(password.encode()).hexdigest()
        with connection.cursor() as cursor:
            cursor.execute("SELECT password FROM admin WHERE name = %s", (username,))
            result = cursor.fetchone()
            if result and result[0] == hashed_password:
                return True
            else:
                return False
    except Exception as e:
        print("An error occurred while authenticating:", e)
        return False
    finally:
        connection.close()

# Insert video information into the database
def insert_video(connection, video_title, video_url):
    try:
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO videos (title, url) VALUES (%s, %s)", (video_title, video_url))
            connection.commit()
            cursor.execute("SELECT LAST_INSERT_ID()")
            video_id = cursor.fetchone()[0]
            return video_id
    except Exception as e:
        print("An error occurred while inserting the video:", e)
        return None

# Insert chunk information into the database
def insert_chunk_into_database(connection, chunk_path, video_id):
    try:
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO chunks (chunkpath, video_id) VALUES (%s, %s)", (chunk_path, video_id))
            connection.commit()
    except Exception as e:
        print("An error occurred while inserting the chunk:", e)

# Insert chunks into the database
def insert_chunks_into_database(connection, chunks_folder_path, video_id):
    last_chunk_path = None
    for chunk_filename in os.listdir(chunks_folder_path):
        chunk_path = os.path.join(chunks_folder_path, chunk_filename)
        insert_chunk_into_database(connection, chunk_path, video_id)
        last_chunk_path = chunk_path
    return last_chunk_path

# Split MP3 into chunks
def split_mp3_to_chunks(mp3_file, chunk_length_sec, output_path, title):
    try:
        mp3_path = os.path.join(output_path, mp3_file)
        audio = AudioFileClip(mp3_path)
        audio_duration_sec = audio.duration
        num_chunks = int(audio_duration_sec / chunk_length_sec) + 1
        folder_path = os.path.join(output_path, title)
        os.makedirs(folder_path, exist_ok=True)
        for i in range(num_chunks):
            start_time = i * chunk_length_sec
            end_time = min((i + 1) * chunk_length_sec, audio_duration_sec)
            chunk = audio.subclip(start_time, end_time)
            chunk.write_audiofile(os.path.join(folder_path, f"chunk_{i+1}.mp3"))
        print("MP3 file split into chunks successfully!")
        return folder_path
    except Exception as e:
        print("An error occurred:", e)

# Convert MP4 to MP3
def convert_mp4_to_mp3(mp4_file, output_path, title, url):
    try:
        mp4_path = os.path.join(output_path, mp4_file)
        clip = VideoFileClip(mp4_path)
        audio = clip.audio
        mp3_file = os.path.splitext(mp4_file)[0] + ".mp3"
        mp3_path = os.path.join(output_path, os.path.basename(mp3_file))
        audio.write_audiofile(mp3_path)
        clip.close()
        os.remove(mp4_path)
        print("MP4 converted to MP3 successfully!")
        path = split_mp3_to_chunks(mp3_file, 10, output_path, title)
        os.remove(mp3_path)
        connection = connect_to_mysql()
        video_id = insert_video(connection, title, url)
        last_chunk_path = insert_chunks_into_database(connection, path, video_id)
        connection.close()
        return last_chunk_path
    except Exception as e:
        print("An error occurred:", e)

# Download YouTube video
def download_youtube_video(url, output_path):
    try:
        output_path = output_path.replace(" ", "_")
        os.makedirs(output_path, exist_ok=True)
        yt = YouTube(url)
        video = yt.streams.first()
        video.download(output_path=output_path)
        print("Video downloaded successfully!")
        return convert_mp4_to_mp3(yt.title+".mp4", output_path, yt.title, url)
    except Exception as e:
        print("An error occurred:", e)

# Main function
script_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.join(script_directory, "videos")


def admin_page():
    st.title("YouTube Video Downloader and Processor")
    
    # Login form
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state['authenticated'] = True
                st.success("Login successful!")
                st.experimental_set_query_params(page="admin")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")
    else:
        # Admin panel
        st.sidebar.title("Admin Panel")
        st.sidebar.write("Welcome to the admin page!")
        
        # Add a logout button
        if st.sidebar.button("Logout"):
            st.session_state['authenticated'] = False
            st.experimental_set_query_params(page="main")
            st.experimental_rerun()

        st.header("Download and Process Video")
        url = st.text_input("Enter YouTube URL:")
        if st.button("Download and Process"):
            if url:
                st.write("Downloading and processing...")
                last_chunk_path = download_youtube_video(url, project_directory)
                st.success("Video downloaded and processed successfully!")
                st.write("Path of the last chunk:", last_chunk_path)
            else:
                st.warning("Please enter a valid YouTube URL!")

if __name__ == "__main__":
    admin_page()
