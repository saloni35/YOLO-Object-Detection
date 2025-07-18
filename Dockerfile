# Use a slim Python image as the base.
# 'slim-buster' is a good choice for smaller image size while providing necessary tools.
# Adjust the Python version (e.g., 3.8, 3.9, 3.10, 3.11) to match what your application uses.
FROM python:3.13.2-slim-bullseye

 # Default to GPU, or 'cpu'
ARG BUILD_TYPE=gpu

# Set the working directory inside the container.
# All subsequent commands will be executed relative to this directory.
WORKDIR /app

# Install system-level dependencies required for some common object detection libraries.
# For example, OpenCV often needs these.
# 'apt-get update' updates the package list.
# 'apt-get install -y' installs packages without prompting.
# '--no-install-recommends' helps keep the image smaller.
# Clean up apt caches to further reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    libwebp-dev \
    libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container.
# We copy this first to leverage Docker's build cache.
# If requirements.txt doesn't change, Docker won't re-run the pip install step.
COPY requirements.txt .
COPY requirements_cpu.txt .

# Install Python dependencies from requirements.txt.
# '--no-cache-dir' prevents pip from storing cache, reducing image size.
# 'upgrade pip' ensures you're using a recent version of pip.
RUN pip install --upgrade pip

RUN if [ "$BUILD_TYPE" = "cpu" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements_cpu.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy the rest of your application code into the container.
# The '.' indicates copying from the current directory on your host to the WORKDIR in the container.

COPY . .

# If your application is a web server or API, expose the port it listens on.
# Replace 8000 with the actual port your application uses (e.g., 5000, 8080).
# If your application is purely a command-line script, you can remove this line.
EXPOSE 8000

# Define the command to run your application when the container starts.
# This assumes your main script is 'app.py'. Adjust if your entry point is different.
# For example:
CMD python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

