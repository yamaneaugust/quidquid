#!/bin/bash
# Startup script for Railway deployment with static file serving

# Copy static files to Streamlit's static directory
mkdir -p ~/.streamlit/static
cp static/robots.txt static/sitemap.xml ~/.streamlit/static/ 2>/dev/null || true

# Start Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
