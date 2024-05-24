import streamlit as st
import pymysql
import testModel  # Import the testModel module
import admin  # Import the admin module

def connect_to_mysql():
    host = "localhost"
    username = "root"
    password = "root"
    database = "streamlit"
    connection = pymysql.connect(host=host, user=username, password=password, database=database)
    return connection

def get_chunk(connection):
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM chunks WHERE transcription IS NULL ORDER BY RAND() LIMIT 1"
            cursor.execute(query)
            chunk = cursor.fetchone()
            return chunk
    except Exception as e:
        st.error(f"An error occurred while fetching chunk: {e}")
        return None

def update_transcription(connection, chunk_id, transcription):
    try:
        with connection.cursor() as cursor:
            query = "UPDATE chunks SET transcription = %s WHERE id = %s"
            cursor.execute(query, (transcription, chunk_id))
            connection.commit()
            st.success("Transcription updated successfully!")
    except Exception as e:
        st.error(f"An error occurred while updating transcription: {e}")

def main_page():
    st.markdown("""
        <style>
            audio::-webkit-media-controls-panel,
            audio::-webkit-media-controls-enclosure {
                background-color: rgb(33, 194, 170);
                drop-shadow: 1px 1px 10px grey;
                border-radius: 30px;
                border: 2px solid rgb(165, 203, 198);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title-text'>Audio Chunk</h1>", unsafe_allow_html=True)

    connection = connect_to_mysql()
    chunk = get_chunk(connection)
    
    if chunk:
        st.audio(chunk[1], format='audio/mp3')
        
        transcription = st.text_input("Enter transcription:")
        if st.button("Submit"):
            update_transcription(connection, chunk[0], transcription)
    else:
        st.warning("No chunks found. Please check back later.")
    connection.close()

    if st.button("Test Model"):
        st.experimental_set_query_params(page="test")

def main():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["main"])[0]

    if page == "main":
        main_page()
    elif page == "test":
        testModel.test_page()
    elif page == "admin":
        admin.admin_page()

if __name__ == "__main__":
    main()
