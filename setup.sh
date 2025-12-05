#!/bin/bash
# Railway setup script to install OpenGL libraries

echo "Installing OpenGL libraries for cv2..."

# Try to install using apt-get (if available)
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y \
        libgl1-mesa-dri \
        libgl1 \
        libglx-mesa0 \
        libglx0 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 || echo "apt-get install failed, continuing..."
fi

# Ensure opencv-python-headless is used
echo "Ensuring opencv-python-headless is installed..."
pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
pip install opencv-python-headless --force-reinstall

echo "Setup complete!"
