# Quick Setup Guide for Windows

## Where is your quidquid project?

You need to either:
1. Find where you cloned the repository
2. OR clone it fresh to your machine

---

## Option A: Find Your Existing Project

Search for the quidquid folder:

```powershell
# In PowerShell, search common locations
Get-ChildItem -Path C:\Users\yaman -Recurse -Directory -Filter "quidquid" -ErrorAction SilentlyContinue
```

Once found, navigate to it:
```powershell
cd path\to\quidquid  # Replace with actual path
```

---

## Option B: Clone/Download the Repository Fresh (RECOMMENDED)

### If you have Git installed:

```powershell
# Navigate to where you want the project
cd C:\Users\yaman\Documents  # Or Desktop, or wherever you prefer

# Clone the repository
git clone <your-repository-url> quidquid

# Navigate into it
cd quidquid
```

### If you DON'T have Git:

1. **Download the repository as a ZIP**:
   - Go to your repository on GitHub/GitLab
   - Click "Code" → "Download ZIP"
   - Extract to: `C:\Users\yaman\Documents\quidquid`

2. **Navigate to it**:
   ```powershell
   cd C:\Users\yaman\Documents\quidquid
   ```

---

## Option C: Create the Project from Scratch Locally

If you want to work locally without cloning:

```powershell
# Create project directory
New-Item -ItemType Directory -Path "C:\Users\yaman\Documents\quidquid"

# Navigate to it
cd C:\Users\yaman\Documents\quidquid

# Download project files from the repository
# (You'll need to manually download or copy the files)
```

---

## Quick Setup Commands (After you navigate to the project)

Once you're in the quidquid directory:

```powershell
# Check you're in the right place
dir  # Should show: src/, scripts/, data/, README.md, etc.

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Extract your ground truth CSV
python scripts\organize_isic_data.py --extract-csv "C:\Users\yaman\Downloads\ISIC2018_Task3_Training_GroundTruth.zip"
```

---

## Recommended Location

I suggest creating the project in:
- `C:\Users\yaman\Documents\quidquid`
- OR `C:\Users\yaman\Desktop\quidquid`
- OR `C:\Users\yaman\projects\quidquid`

---

## Quick Copy-Paste Setup

```powershell
# Create and navigate to project location
New-Item -ItemType Directory -Path "C:\Users\yaman\Documents\quidquid" -Force
cd C:\Users\yaman\Documents\quidquid

# If you have the files from git, they should be here
# If not, you need to download/clone the repository first
```

---

## Need the Repository URL?

If this is a GitHub/GitLab project, you need the clone URL.

Based on the git remote we saw earlier:
- Repository: `yamaneaugust/quidquid`

You would clone with:
```powershell
git clone https://github.com/yamaneaugust/quidquid.git
# OR
git clone <your-git-url>
```

---

## Can't Find Git?

Install Git for Windows:
1. Download: https://git-scm.com/download/win
2. Install with default options
3. Restart PowerShell
4. Then run the clone command

---

## Alternative: Download Files Manually

If you have access to the repository through a web interface:
1. Go to the repository website
2. Click "Code" or "Download"
3. Download as ZIP
4. Extract to `C:\Users\yaman\Documents\quidquid`
5. Then run the setup commands above
