# Oracle Cloud Free Tier Deployment Guide
## Deploy Modium on Oracle Cloud - Forever Free

This guide will help you deploy Modium on Oracle Cloud's Free Tier, allowing you to:
- Use modium.io as your URL (no more streamlit.app redirect)
- Host for free forever (not a trial)
- Have full control over your deployment
- Keep the app running 24/7

**Estimated Time:** 1-2 hours

---

## Prerequisites

- Oracle Cloud account (we'll create this)
- Porkbun account with modium.io domain (you already have this)
- SSH client (built into Windows 10+, Mac, Linux)
- Basic terminal/command line familiarity

---

## Phase 1: Oracle Cloud Account Setup

### Step 1: Create Oracle Cloud Account

1. Go to https://www.oracle.com/cloud/free/
2. Click "Start for free"
3. Fill in your information:
   - Email address
   - Country/Region
   - Name
4. You'll need to verify with:
   - Phone number
   - Credit card (for verification only - you won't be charged)
5. Choose "Free Tier Account" (NOT the paid option)
6. Complete verification

**IMPORTANT:** Oracle requires a credit card for verification but will NOT charge you if you stay within Free Tier limits. We'll set up budget alerts to be safe.

### Step 2: Set Up Budget Alerts

Once logged in:

1. Click hamburger menu (≡) → **Billing & Cost Management**
2. Click **Budgets**
3. Click **Create Budget**
4. Set:
   - Monthly Budget: $1.00
   - Alert at: 50%, 80%, 100%
   - Email: Your email
5. Click **Create**

This ensures you'll be notified if anything tries to charge you.

---

## Phase 2: Create a VM Instance

### Step 1: Launch Instance

1. Click hamburger menu (≡) → **Compute** → **Instances**
2. Click **Create Instance**

### Step 2: Configure Instance

**Name:** `modium-app`

**Placement:**
- Keep defaults (Availability Domain: AD-1 or similar)

**Image and Shape:**

1. Click **Change Image**
   - Select **Canonical Ubuntu** (latest version, e.g., 22.04)
   - Click **Select Image**

2. Click **Change Shape**
   - Select **Ampere** (ARM-based)
   - Choose **VM.Standard.A1.Flex**
   - Set:
     - OCPUs: **2**
     - Memory (GB): **12**
   - Click **Select Shape**

**Why these specs?** Oracle's Free Tier gives you up to 4 OCPUs and 24GB RAM total across all ARM instances. We're using half for Modium, leaving room for future projects.

### Step 3: Networking

**Virtual Cloud Network:**
- Keep default (automatically created)

**Subnet:**
- Keep default (Public Subnet)

**Public IP:**
- Select **Assign a public IPv4 address** ✓

### Step 4: SSH Keys

**CRITICAL STEP - Save these keys carefully!**

**Option A: Generate New Keys (Recommended)**

1. Select **Generate a key pair for me**
2. Click **Save Private Key** → Save as `modium-key.pem`
3. Click **Save Public Key** → Save as `modium-key.pub`
4. Store these in a safe location (you'll need them to access your server)

**Option B: Use Existing Keys**

If you already have SSH keys, you can upload your public key instead.

### Step 5: Boot Volume

- Keep defaults (50GB boot volume)
- This is plenty for Modium

### Step 6: Create

1. Review all settings
2. Click **Create**
3. Wait 1-2 minutes for provisioning

**Your VM is now being created!**

---

## Phase 3: Configure Firewall Rules

By default, Oracle Cloud blocks all traffic except SSH (port 22). We need to open ports 80 (HTTP) and 443 (HTTPS).

### Step 1: Find Your VCN

1. Click hamburger menu (≡) → **Networking** → **Virtual Cloud Networks**
2. Click on your VCN (should be named like `vcn-YYYYMMDD-HHMM`)

### Step 2: Configure Security List

1. Click **Security Lists** (left sidebar)
2. Click on **Default Security List for vcn-...**
3. Click **Add Ingress Rules**

**Rule 1 - HTTP (port 80):**
- Source CIDR: `0.0.0.0/0`
- IP Protocol: TCP
- Source Port Range: (leave blank)
- Destination Port Range: `80`
- Description: `HTTP traffic`
- Click **Add Ingress Rules**

**Rule 2 - HTTPS (port 443):**
- Repeat above but change:
  - Destination Port Range: `443`
  - Description: `HTTPS traffic`

### Step 3: Configure OS Firewall

We'll do this after SSH'ing into the server (next step).

---

## Phase 4: Connect to Your Server

### Step 1: Get Your Server's IP Address

1. Go to **Compute** → **Instances**
2. Click on **modium-app**
3. Copy the **Public IP Address** (e.g., 123.456.789.012)

### Step 2: Set Key Permissions

**On Mac/Linux:**
```bash
chmod 400 /path/to/modium-key.pem
```

**On Windows (PowerShell):**
```powershell
icacls "C:\path\to\modium-key.pem" /inheritance:r
icacls "C:\path\to\modium-key.pem" /grant:r "%username%:R"
```

### Step 3: SSH Into Server

**Mac/Linux:**
```bash
ssh -i /path/to/modium-key.pem ubuntu@YOUR_IP_ADDRESS
```

**Windows (PowerShell or CMD):**
```powershell
ssh -i C:\path\to\modium-key.pem ubuntu@YOUR_IP_ADDRESS
```

Replace `YOUR_IP_ADDRESS` with the actual IP from Step 1.

**First time:** You'll see a message about authenticity. Type `yes` and press Enter.

**You're now connected to your server!**

---

## Phase 5: Server Setup

Run these commands one by one in your SSH session.

### Step 1: Update System

```bash
sudo apt update && sudo apt upgrade -y
```

This takes 2-3 minutes.

### Step 2: Install Python 3.10+

```bash
sudo apt install -y python3 python3-pip python3-venv git
```

### Step 3: Configure OS Firewall

Ubuntu uses `iptables` which Oracle's default image has strict rules. We need to allow HTTP/HTTPS:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

If `netfilter-persistent` isn't installed:
```bash
sudo apt install -y iptables-persistent
# Then re-run the save command
```

### Step 4: Install Nginx (Reverse Proxy)

```bash
sudo apt install -y nginx
```

### Step 5: Clone Your Repository

```bash
cd ~
git clone https://github.com/yamaneaugust/quidquid.git
cd quidquid
```

### Step 6: Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now show `(venv)`.

### Step 7: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes 5-10 minutes (PyTorch is large).

---

## Phase 6: Download Model File

Your model is on Google Drive. We need to download it to the server.

### Option A: Using gdown (Recommended)

```bash
pip install gdown

# Download model
cd ~/quidquid/data/models/
gdown 1ybi3JF3gWAlahmN3h-pxd-oQGTzxE77K -O best_model.pth
```

### Option B: Manual Upload via SCP

If gdown doesn't work, upload from your local machine:

**On your local computer (not SSH session):**

**Mac/Linux:**
```bash
scp -i /path/to/modium-key.pem \
    /path/to/local/best_model.pth \
    ubuntu@YOUR_IP_ADDRESS:~/quidquid/data/models/
```

**Windows (PowerShell):**
```powershell
scp -i C:\path\to\modium-key.pem `
    C:\path\to\local\best_model.pth `
    ubuntu@YOUR_IP_ADDRESS:~/quidquid/data/models/
```

---

## Phase 7: Configure Streamlit for Production

### Step 1: Create Streamlit Config

```bash
mkdir -p ~/.streamlit
nano ~/.streamlit/config.toml
```

Paste this content:

```toml
[server]
headless = true
address = "127.0.0.1"
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
serverAddress = "modium.io"
gatherUsageStats = false
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Step 2: Create Startup Script

```bash
nano ~/start_modium.sh
```

Paste:

```bash
#!/bin/bash
cd ~/quidquid
source venv/bin/activate
streamlit run app.py
```

Make it executable:

```bash
chmod +x ~/start_modium.sh
```

---

## Phase 8: Set Up Systemd Service (Auto-Start)

This makes Modium start automatically when the server boots.

### Step 1: Create Service File

```bash
sudo nano /etc/systemd/system/modium.service
```

Paste:

```ini
[Unit]
Description=Modium AI Skin Lesion Screener
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/quidquid
Environment="PATH=/home/ubuntu/quidquid/venv/bin"
ExecStart=/home/ubuntu/quidquid/venv/bin/streamlit run app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save with `Ctrl+X`, `Y`, `Enter`.

### Step 2: Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable modium
sudo systemctl start modium
```

### Step 3: Check Status

```bash
sudo systemctl status modium
```

You should see **active (running)** in green.

**Common commands:**
- View logs: `sudo journalctl -u modium -f`
- Restart: `sudo systemctl restart modium`
- Stop: `sudo systemctl stop modium`

---

## Phase 9: Configure Nginx Reverse Proxy

Nginx will forward traffic from modium.io (port 80/443) to Streamlit (port 8501).

### Step 1: Create Nginx Config

```bash
sudo nano /etc/nginx/sites-available/modium
```

Paste:

```nginx
server {
    listen 80;
    server_name modium.io www.modium.io;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Save with `Ctrl+X`, `Y`, `Enter`.

### Step 2: Enable Site

```bash
sudo ln -s /etc/nginx/sites-available/modium /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

---

## Phase 10: Configure DNS (Porkbun)

Now we point modium.io to your Oracle Cloud server.

### Step 1: Get Your Server IP

If you forgot it:
```bash
curl ifconfig.me
```

This shows your public IP (e.g., 123.456.789.012).

### Step 2: Update DNS in Porkbun

1. Log into https://porkbun.com
2. Go to **Domain Management**
3. Click **DNS** next to modium.io
4. **Delete** any existing A records (if present)
5. Click **Add Record**

**Record 1 - Root Domain:**
- Type: `A`
- Host: (leave blank or `@`)
- Answer: `YOUR_SERVER_IP`
- TTL: `600`

**Record 2 - www Subdomain:**
- Type: `A`
- Host: `www`
- Answer: `YOUR_SERVER_IP`
- TTL: `600`

6. Click **Save** for each

**DNS propagation takes 5-60 minutes.** You can check status at https://dnschecker.org

---

## Phase 11: Set Up SSL Certificate (HTTPS)

Use Let's Encrypt for free SSL certificates.

### Step 1: Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### Step 2: Get Certificate

```bash
sudo certbot --nginx -d modium.io -d www.modium.io
```

**During setup:**
- Email: Your email (for renewal notices)
- Terms: Press `A` to agree
- Share email: Press `N`
- Redirect: Press `2` (redirect HTTP to HTTPS)

**Certbot will:**
- Obtain certificates
- Update Nginx config
- Set up auto-renewal

### Step 3: Test Auto-Renewal

```bash
sudo certbot renew --dry-run
```

Should show "Congratulations, all simulated renewals succeeded".

---

## Phase 12: Verify Deployment

### Step 1: Test HTTP → HTTPS Redirect

```bash
curl -I http://modium.io
```

Should show `301 Moved Permanently` and `Location: https://modium.io/`

### Step 2: Visit Your Site

Open browser and go to: **https://modium.io**

You should see Modium loading!

### Step 3: Check Certificate

Click the padlock icon in browser. Should show:
- Valid certificate
- Issued by Let's Encrypt
- Valid for modium.io and www.modium.io

---

## Phase 13: Optimization & Monitoring

### Increase Upload Size Limit

Streamlit might reject large images. Fix:

```bash
sudo nano /etc/nginx/nginx.conf
```

Find the `http` block and add:

```nginx
http {
    ...
    client_max_body_size 10M;  # Add this line
    ...
}
```

Save and restart:

```bash
sudo nginx -t
sudo systemctl restart nginx
```

### Set Up Log Rotation

```bash
sudo nano /etc/logrotate.d/modium
```

Paste:

```
/home/ubuntu/quidquid/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### Monitor System Resources

```bash
# Check memory usage
free -h

# Check disk usage
df -h

# Check running processes
htop  # Install with: sudo apt install htop
```

### Enable Automatic Security Updates

```bash
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

Select "Yes" to enable.

---

## Maintenance Commands

### Update Modium Code

```bash
cd ~/quidquid
git pull origin claude/cnn-lesion-screening-011CUM48AsuFKri5BSwi1P5r
sudo systemctl restart modium
```

### View Logs

```bash
# Streamlit logs
sudo journalctl -u modium -f

# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Nginx error logs
sudo tail -f /var/log/nginx/error.log
```

### Check Service Status

```bash
# Modium app
sudo systemctl status modium

# Nginx
sudo systemctl status nginx

# Check if Streamlit is listening on port 8501
sudo netstat -tulpn | grep 8501
```

### Restart Everything

```bash
sudo systemctl restart modium
sudo systemctl restart nginx
```

---

## Troubleshooting

### Site not loading?

**Check DNS propagation:**
```bash
nslookup modium.io
```

Should show your Oracle Cloud IP.

**Check if Streamlit is running:**
```bash
sudo systemctl status modium
```

If it's failed:
```bash
sudo journalctl -u modium -n 50
```

**Check Nginx:**
```bash
sudo nginx -t
sudo systemctl status nginx
```

### 502 Bad Gateway?

Streamlit isn't running or crashed.

```bash
# Check logs
sudo journalctl -u modium -f

# Restart service
sudo systemctl restart modium
```

### Upload images not working?

Check file size limit in Nginx config (Phase 13).

### SSL certificate issues?

```bash
# Check certificate status
sudo certbot certificates

# Renew manually
sudo certbot renew
```

### Database not persisting?

Make sure data directory exists:

```bash
mkdir -p ~/quidquid/data
sudo chown ubuntu:ubuntu ~/quidquid/data
sudo systemctl restart modium
```

### Out of memory?

Check usage:
```bash
free -h
```

If low, reduce Streamlit workers or upgrade to more RAM (still free tier):
```bash
# Stop instance
# Change shape to 4 OCPUs / 24GB RAM
# Start instance
```

---

## Cost Management

### Monitor Your Usage

1. Log into Oracle Cloud Console
2. Go to **Billing & Cost Management**
3. Check **Cost Analysis**

**You should always see $0.00 if staying in Free Tier.**

### What's Free Forever?

- 2 VM.Standard.A1.Flex instances (up to 4 OCPUs, 24GB RAM total)
- 2 Block Volumes (200GB total)
- 10GB Object Storage
- 10TB outbound data transfer per month
- Load Balancer (1 instance, 10 Mbps)

**Modium uses:**
- 1 VM instance (2 OCPUs, 12GB RAM)
- 1 Block Volume (50GB boot disk)
- Minimal outbound transfer (unless you get massive traffic)

**You're well within limits!**

---

## Performance Tuning

### If Modium is slow:

**Option 1: Use full Free Tier allocation**
- Upgrade VM to 4 OCPUs / 24GB RAM (still free)

**Option 2: Add caching to app.py**
```python
import streamlit as st

@st.cache_resource
def load_model():
    # Your model loading code
    pass
```

**Option 3: Enable gzip in Nginx**

Already done, but verify in `/etc/nginx/nginx.conf`:
```nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript;
```

---

## Security Checklist

- [ ] SSH key-based authentication (password auth disabled by default)
- [ ] Firewall configured (only ports 22, 80, 443 open)
- [ ] SSL/TLS enabled (HTTPS)
- [ ] Auto-security updates enabled
- [ ] Budget alerts configured
- [ ] Regular backups (manual for now)

### Optional: Set Up Backups

**Backup database:**
```bash
# Create backup script
nano ~/backup_modium.sh
```

Paste:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp ~/quidquid/data/users.db ~/backups/users_${DATE}.db
# Keep only last 7 backups
ls -t ~/backups/users_*.db | tail -n +8 | xargs rm -f
```

Make executable and set up cron:
```bash
chmod +x ~/backup_modium.sh
mkdir ~/backups
crontab -e
```

Add line:
```
0 3 * * * /home/ubuntu/backup_modium.sh
```

This backs up database daily at 3 AM.

---

## Success Checklist

- [ ] Oracle Cloud account created
- [ ] VM instance running (modium-app)
- [ ] SSH access working
- [ ] Modium code deployed
- [ ] Model file downloaded
- [ ] Streamlit running as systemd service
- [ ] Nginx reverse proxy configured
- [ ] DNS pointing to Oracle Cloud IP
- [ ] SSL certificate installed
- [ ] https://modium.io loads correctly
- [ ] Budget alerts configured
- [ ] Auto-updates enabled

---

## Next Steps

Once deployment is complete:

1. **Update Google Analytics** - Verify tracking works with new domain
2. **Update Product Hunt** - Post an update that you're now self-hosted
3. **Test all features** - Login, signup, analysis, history, comparison
4. **Monitor for 24 hours** - Check logs and ensure stability
5. **Remove Streamlit Cloud deployment** - No longer needed
6. **Update GitHub README** - New deployment URL

---

## Getting Help

**Oracle Cloud Issues:**
- Docs: https://docs.oracle.com/en-us/iaas/
- Community: https://community.oracle.com/

**Streamlit Issues:**
- Docs: https://docs.streamlit.io/
- Forum: https://discuss.streamlit.io/

**Nginx Issues:**
- Docs: https://nginx.org/en/docs/

**Let's Encrypt Issues:**
- Docs: https://certbot.eff.org/

---

## Estimated Costs

**Setup:** $0 (Free Tier)
**Monthly:** $0 (Forever Free)
**Domain:** $9.13/year (Porkbun - already paid)

**Total Ongoing Cost:** $0.76/month (domain only)

vs. Streamlit Teams at $20/month = **Save $230/year**

---

You now have a production-grade deployment of Modium running on your own infrastructure!

The URL bar will always show `modium.io`, you have full control, and it's completely free.
