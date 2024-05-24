FROM python:3.9

#working directory in the container
WORKDIR /usr/src/app


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run your Streamlit application
CMD ["streamlit", "run", "index.py"]
