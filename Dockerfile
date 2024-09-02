# Use the official slim Python image from the Docker Hub
FROM python:3.11.1-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy only the necessary files into the container at /app
COPY GAN_Architecture.py .
COPY model.py .
COPY preprocess.py .
COPY sampler.py .
COPY flask-app.py .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define the command to run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "1800", "--workers", "2", "flask-app:app"]