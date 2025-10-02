# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required by OpenCV (cv2)
RUN apt-get update && apt-get install -y libgl1 libgthread-2.0-0

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Hugging Face Spaces exposes port 7860 for web apps by default
EXPOSE 7860

# The command to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--worker-tmp-dir", "/dev/shm", "app:app"]