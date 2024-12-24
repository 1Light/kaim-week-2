# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Add any additional packages your app might need here, e.g.:
    # build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that your Streamlit app will run on
EXPOSE 8501

# Set the command to run your app (for example, Streamlit)
CMD ["streamlit", "run", "app/main.py"]