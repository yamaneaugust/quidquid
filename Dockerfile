# Use Python 3.11 slim image based on Debian Bookworm (stable)
# Bookworm still has libgl1-mesa-glx available, unlike Trixie
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other image processing libraries
# Using Bookworm which still has libgl1-mesa-glx available
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
