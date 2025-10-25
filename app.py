"""
Streamlit Web Interface for Lesion Pre-Screening

A minimalist, professional web interface for uploading lesion images
and receiving AI-powered risk assessments.

NOT FOR MEDICAL DIAGNOSIS - Educational purposes only.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import sys
import os
import base64
from pathlib import Path
import gdown

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predict import LesionPredictor
from src.preprocessing.image_processing import load_image
from src.auth.user_db import UserDatabase
from src.auth.auth_ui import show_login_page, show_user_menu, show_history_page

# Page config
st.set_page_config(
    page_title="Modium - AI-Powered Skin Lesion Pre-Screening",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# SEO Meta Tags for Google indexing
st.markdown("""
<meta name="description" content="Modium - AI-powered skin lesion pre-screening tool. Upload images for risk assessment using CNN deep learning and ABCDE criteria analysis. Free educational tool for skin cancer awareness.">
<meta name="keywords" content="skin lesion, melanoma screening, AI dermatology, skin cancer detection, ABCDE criteria, lesion analysis, deep learning, medical AI">
<meta name="author" content="Modium">
<meta name="robots" content="index, follow">
<meta property="og:title" content="Modium - AI Skin Lesion Pre-Screening">
<meta property="og:description" content="Free AI-powered skin lesion risk assessment using deep learning and ABCDE criteria analysis">
<meta property="og:type" content="website">
<meta property="og:url" content="https://modium.io">
""", unsafe_allow_html=True)

# Custom CSS for black minimalist theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
    }

    /* File uploader */
    .uploadedFile {
        background-color: #1a1a1a;
        border: 1px solid #333333;
    }

    /* Buttons */
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        border: none;
        border-radius: 2px;
        font-weight: 500;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #cccccc;
        border: none;
    }

    /* Warning box */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #ff4444;
        color: #ffffff;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    /* Divider */
    hr {
        border-color: #333333;
    }

    /* Text */
    p, li, span {
        color: #cccccc;
    }

    /* Download button */
    .stDownloadButton>button {
        background-color: #ffffff;
        color: #000000;
        border: none;
        border-radius: 2px;
        font-weight: 500;
    }

    /* Remove padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_database():
    """Get database instance (cached)."""
    return UserDatabase()

db = get_database()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

# Check if user needs to login
if st.session_state['authenticated'] is None:
    show_login_page(db)
    st.stop()

# Show user menu in sidebar
if st.session_state.get('authenticated') or st.session_state.get('guest_mode'):
    show_user_menu(db)

# Route to appropriate page
if st.session_state.get('page') == 'history':
    show_history_page(db)
    st.stop()

# Title
st.markdown("<p style='text-align: center; font-size: 1.5rem; color: #888888; margin-bottom: 0.5rem; letter-spacing: 8px; font-weight: 300;'>Modium</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>Lesion Pre-Screening</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888; margin-top: 0;'>AI-Powered Skin Cancer Risk Assessment</p>", unsafe_allow_html=True)

# Critical disclaimer
st.error("""
⚠️ **NOT FOR MEDICAL DIAGNOSIS**
This system is for educational and documentation purposes only.
Always consult a qualified healthcare professional for medical decisions.
""")

st.markdown("---")

# Model selection
@st.cache_resource
def load_model(model_path, class_names):
    """Load the trained model (cached)."""
    try:
        predictor = LesionPredictor(
            model_path=model_path,
            class_names=class_names,
            device='cpu'  # Use CPU for web app
        )
        return predictor
    except Exception as e:
        return None

# Configuration
MODEL_PATH = "data/models/best_model.pth"
CLASS_NAMES = ["benign", "suspicious", "malignant"]
GDRIVE_FILE_ID = "1ybi3JF3gWAlahmN3h-pxd-oQGTzxE77K"

# Auto-download model from Google Drive if not present
if not Path(MODEL_PATH).exists():
    with st.spinner("Downloading AI model from cloud storage (first time only)..."):
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

            # Download from Google Drive
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            st.info("Please contact support or try again later.")
            st.stop()

# Load model
with st.spinner("Loading AI model..."):
    predictor = load_model(MODEL_PATH, CLASS_NAMES)

if predictor is None:
    st.error("Failed to load model. Please check the model file.")
    st.stop()

st.markdown("---")

# File upload
st.markdown("<h3>Upload Lesion Image</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width='stretch')

    st.markdown("---")

    # Analyze button
    if st.button("ANALYZE", use_container_width=True):
        with st.spinner("Analyzing image..."):
            # Save temp file
            temp_path = "temp_upload.jpg"
            image.save(temp_path)

            # Run prediction
            try:
                predictions, visual_features = predictor.predict_image(temp_path)
                risk_score = predictor.assess_risk(predictions, visual_features)

                # Save analysis to database if user is logged in
                if st.session_state.get('authenticated'):
                    # Convert image to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    # Get predicted class
                    predicted_idx = np.argmax(predictions)
                    predicted_class = CLASS_NAMES[predicted_idx]
                    confidence = float(predictions[predicted_idx])

                    # Save to database
                    db.save_analysis(
                        user_id=st.session_state['user_id'],
                        image_base64=img_str,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        risk_score=float(risk_score.overall_score),
                        risk_level=risk_score.risk_level.value,
                        visual_features=visual_features,
                        recommendations=risk_score.recommendations
                    )

                # Display results
                st.markdown("<h2 style='text-align: center;'>RISK ASSESSMENT</h2>", unsafe_allow_html=True)

                # Risk level with color
                risk_colors = {
                    "Low": "#28a745",
                    "Moderate": "#ffc107",
                    "High": "#fd7e14",
                    "Very High": "#dc3545"
                }
                risk_color = risk_colors.get(risk_score.risk_level.value, "#ffffff")

                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background-color: #1a1a1a; border-radius: 4px; margin: 2rem 0;'>
                    <h1 style='color: {risk_color}; font-size: 4rem; margin: 0;'>{risk_score.overall_score:.0f}</h1>
                    <p style='color: #888888; font-size: 1rem; margin: 0;'>RISK SCORE (0-100)</p>
                    <h2 style='color: {risk_color}; margin-top: 1rem;'>{risk_score.risk_level.value.upper()}</h2>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                st.markdown("<h3>Model Predictions</h3>", unsafe_allow_html=True)

                cols = st.columns(3)
                for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                    with cols[i]:
                        st.metric(
                            label=class_name.capitalize(),
                            value=f"{prob:.1%}"
                        )

                st.markdown("---")

                # Visual features (ABCDE criteria)
                st.markdown("<h3>Visual Features (ABCDE Criteria)</h3>", unsafe_allow_html=True)

                features_display = {
                    'asymmetry': 'A - Asymmetry',
                    'border_irregularity': 'B - Border Irregularity',
                    'color_variation': 'C - Color Variation',
                    'diameter_score': 'D - Diameter (Relative Size)'
                }

                for key, label in features_display.items():
                    value = visual_features.get(key, 0)
                    if value is not None:
                        st.progress(value, text=f"{label}: {value:.2f}")

                # Evolution (E) - Special handling since it needs historical data
                evolution_note = visual_features.get('evolution_note', 'N/A')
                evolution_rec = visual_features.get('evolution_recommendation', '')

                st.info(f"""
                **E - Evolution**: {evolution_note}

                {evolution_rec}
                """)

                st.markdown("---")

                # Recommendations
                st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)

                for i, rec in enumerate(risk_score.recommendations, 1):
                    if "NOT intended" in rec:
                        st.error(f"**{rec}**")
                    elif "HIGH" in rec.upper() or "URGENT" in rec.upper():
                        st.warning(f"{i}. {rec}")
                    else:
                        st.info(f"{i}. {rec}")

                st.markdown("---")

                # Generate PDF
                st.markdown("<h3>Download Report</h3>", unsafe_allow_html=True)

                if st.button("GENERATE PDF REPORT", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        try:
                            output_pdf = "temp_report.pdf"
                            predictor.generate_report(
                                image_path=temp_path,
                                risk_score=risk_score,
                                output_path=output_pdf
                            )

                            # Read PDF for download
                            with open(output_pdf, "rb") as f:
                                pdf_bytes = f.read()

                            st.download_button(
                                label="DOWNLOAD PDF REPORT",
                                data=pdf_bytes,
                                file_name="lesion_screening_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                            st.success("✓ PDF report generated successfully!")

                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                            # Show full traceback for debugging
                            import traceback
                            with st.expander("Show error details"):
                                st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

else:
    # Instructions when no file uploaded
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; color: #666666;'>
        <p style='font-size: 1.2rem;'>Upload a lesion image to begin analysis</p>
        <p style='font-size: 0.9rem;'>Supported formats: JPG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; font-size: 0.8rem; padding: 2rem 0;'>
    <p>Lesion Pre-Screening CNN System v0.1.0</p>
    <p>For educational and research purposes only</p>
    <p>Always seek professional medical advice</p>
</div>
""", unsafe_allow_html=True)
