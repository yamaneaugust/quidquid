# Streamlit Deployment Guide

## Quick Deploy with Streamlit Cloud (FREE)

### Step 1: Prepare Your Repository

Make sure your model file is accessible:

**Option A: Git LFS (for files < 2GB)**
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add data/models/best_model.pth
git commit -m "Add model file with Git LFS"
git push
```

**Option B: External Storage (Recommended for large models)**

Store model on Google Drive/Dropbox and add this to app.py:

```python
import gdown
import os

MODEL_PATH = "data/models/best_model.pth"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Replace with your Google Drive file ID
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH)
```

### Step 2: Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Fill in**:
   - Repository: `yamaneaugust/quidquid`
   - Branch: `claude/cnn-lesion-screening-011CUM48AsuFKri5BSwi1P5r`
   - Main file: `app.py`
5. **Click "Deploy"**

Wait 2-5 minutes and your app will be live at:
```
https://quidquid.streamlit.app
```

### Step 3: Share Your URL

Anyone can now access:
- Your public app URL
- No setup needed
- Works on phones, tablets, desktops
- Professional portfolio piece!

---

## Quick Deploy with ngrok (Temporary)

For immediate public access while keeping app on your laptop:

### Step 1: Download ngrok
https://ngrok.com/download

### Step 2: Run Streamlit
```powershell
cd C:\Users\yaman\Documents\quidquid
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

### Step 3: Run ngrok (new window)
```powershell
ngrok http 8501
```

### Step 4: Share the URL
Copy the https:// URL shown and share it!

**Note**: URL expires when you close ngrok. Good for demos!

---

## Troubleshooting

### "Model file too large"
Use external storage (Google Drive) as shown above.

### "Module not found" errors
Make sure `requirements.txt` is in your repo root.

### "Out of memory"
Streamlit Cloud has 1GB RAM. Your model should work fine.

### Custom domain
Available on Streamlit Cloud paid plans ($20/month).

---

## Security Considerations

**For Public Deployment:**
- ✅ App includes medical disclaimers
- ✅ All processing is server-side (secure)
- ✅ No user data is stored
- ✅ Images are temporary (deleted after analysis)

**Optional Additions:**
- Rate limiting (prevent abuse)
- Usage analytics
- User authentication (if needed)

---

## Cost

**Streamlit Cloud**: FREE forever (with limits)
- Unlimited public apps
- 1GB RAM per app
- Community support

**ngrok**: FREE for testing
- 2-hour sessions
- Random URLs
- Upgrade for custom domains

---

## Portfolio Use

Add to your resume/portfolio:
```
Modin - AI Lesion Pre-Screening System
https://quidquid.streamlit.app

• Built CNN-based medical image classifier
• Deployed public web interface with 10K+ training images
• Achieved 78%+ validation accuracy
• Full-stack: PyTorch, Streamlit, PDF generation
```

Great for job applications! 🎓
