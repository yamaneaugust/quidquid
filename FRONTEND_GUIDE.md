# Frontend Guide - Lesion Pre-Screening Web Interface

Two minimalist black-themed web interfaces are available for your CNN lesion screening system.

---

## Option 1: Streamlit (Recommended - Easiest)

### Pros:
- ✅ Zero HTML/CSS/JS knowledge required
- ✅ Runs in one command
- ✅ Auto-reload on code changes
- ✅ Built-in file upload and UI components
- ✅ Perfect for demos and prototyping

### Run It:

```powershell
# Install streamlit (if not already installed)
pip install streamlit

# Run the app
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

**That's it!** 🎉

---

## Option 2: Flask (More Professional)

### Pros:
- ✅ Full control over HTML/CSS/JS
- ✅ More professional/production-ready
- ✅ Better for custom branding
- ✅ Can deploy to traditional web servers

### Run It:

```powershell
# Install flask (if not already installed)
pip install flask

# Run the app
python app_flask.py
```

Open your browser to `http://localhost:5000`

---

## Features (Both Apps)

### Core Functionality:
- 📤 **Upload lesion images** (JPG, PNG)
- 🤖 **AI-powered analysis** using your trained CNN
- 📊 **Risk assessment** (0-100 score + Low/Moderate/High/Very High)
- 🎯 **Model predictions** for each class
- 📈 **Visual features** (ABCDE criteria: Asymmetry, Border, Color)
- 📝 **Clinical recommendations** based on risk level
- 📄 **PDF report generation** for documentation

### Design:
- 🖤 **Minimalist black theme** (as requested!)
- 📱 **Mobile responsive**
- ⚡ **Fast and lightweight**
- 🔒 **Privacy-focused** (local processing)

---

## Screenshots/Preview

### Streamlit Version:
- Clean, modern interface
- Sidebar for configuration
- Auto-formatting
- Built-in theming

### Flask Version:
- Custom black minimalist design
- Drag-and-drop upload
- Smooth animations
- Professional look

---

## Usage Workflow

1. **Start the app** (see commands above)
2. **Upload a lesion image**
3. **Click "ANALYZE"**
4. **View risk assessment** and predictions
5. **Review recommendations**
6. **Generate PDF report** (optional)
7. **Download** for documentation

---

## Configuration

### Change Model Path:

Both apps load from `data/models/best_model.pth` by default.

**Streamlit** (`app.py` line 100):
```python
MODEL_PATH = "data/models/best_model.pth"
CLASS_NAMES = ["benign", "suspicious", "malignant"]
```

**Flask** (`app_flask.py` line 21):
```python
MODEL_PATH = "data/models/best_model.pth"
CLASS_NAMES = ["benign", "suspicious", "malignant"]
```

### Change Port:

**Streamlit**:
```powershell
streamlit run app.py --server.port 8080
```

**Flask** (`app_flask.py` line 107):
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## Requirements

### Model File:
You need a trained model at `data/models/best_model.pth`

If you don't have one yet:
```powershell
python -m src.model.train --data-dir data/raw_simplified --num-classes 3 --epochs 30
```

### Dependencies:
```powershell
# For Streamlit
pip install streamlit

# For Flask
pip install flask

# Both are already in requirements.txt
pip install -r requirements.txt
```

---

## Deployment

### Streamlit Cloud (Free):
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo
4. Deploy! (automatic)

### Flask Deployment:
- **Heroku**: `git push heroku main`
- **DigitalOcean**: App Platform or Droplet
- **AWS**: Elastic Beanstalk or EC2
- **Vercel/Netlify**: Serverless functions

---

## Troubleshooting

### "Model not found" Error:
Make sure you have a trained model:
```powershell
dir data\models\best_model.pth
```

If not, train one first (see above).

### Port Already in Use:
**Streamlit**:
```powershell
streamlit run app.py --server.port 8502
```

**Flask**:
Change port in `app_flask.py` line 107

### Slow Performance:
The apps run on CPU by default (safe for web deployment).

For faster inference, edit the predictor initialization to use GPU:
```python
predictor = LesionPredictor(
    model_path=MODEL_PATH,
    class_names=CLASS_NAMES,
    device='cuda'  # Change from 'cpu' to 'cuda'
)
```

### Upload Size Limit:
**Streamlit**: Default 200MB

**Flask**: Set in `app_flask.py` line 16:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

---

## Comparison

| Feature | Streamlit | Flask |
|---------|-----------|-------|
| **Setup** | 1 command | Requires HTML/CSS/JS files |
| **Customization** | Limited | Full control |
| **Learning Curve** | Easy | Moderate |
| **Deployment** | Streamlit Cloud (free) | Traditional hosting |
| **Performance** | Good | Excellent |
| **Best For** | Demos, MVPs | Production apps |

---

## My Recommendation

**Start with Streamlit** - it's running in literally one command:

```powershell
streamlit run app.py
```

Then, if you need more customization later, switch to Flask.

---

## Testing

### Test Images:

Use images from your test dataset:
```powershell
# Streamlit - just upload through the UI

# Flask - same, upload through the UI

# Or test programmatically
python -m src.inference.predict --image "C:\Users\yaman\Downloads\extracted\ISIC2018_Task3_Training_Input\ISIC_0024306.jpg" --model data\models\best_model.pth --output test.pdf --classes benign suspicious malignant
```

---

## Next Steps

1. ✅ **Run the app** (choose Streamlit or Flask)
2. ✅ **Upload a test image**
3. ✅ **Verify it works**
4. ✅ **Customize** branding/colors if desired
5. ✅ **Deploy** to share with others

---

## Support

Both apps include:
- ⚠️ **Medical disclaimers** (NOT for diagnosis)
- 📊 **Full risk assessment pipeline**
- 📄 **PDF report generation**
- 🎨 **Black minimalist design**

Enjoy your sleek new frontend! 🚀
