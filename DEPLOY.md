# Deployment Guide

## Architecture

```
Browser → Cloudflare CDN → Cloudflare Tunnel → Azure VM (localhost:8050)
                ↓
        Cloudflare Access
        (email verification)
```

- Dashboard 只监听 localhost:8050，不暴露任何端口
- 所有流量经 Cloudflare Tunnel 加密传输
- 访问需要 Cloudflare Access 邮箱验证
- 新交易通过 Slack Webhook 通知

## Step 1: Server Setup

SSH 登录 Azure VM:

```bash
# 1. Clone repo
git clone https://github.com/xzjh/crypto-daytrading.git
cd crypto-daytrading

# 2. Install Python dependencies
sudo apt update && sudo apt install -y python3-pip
pip3 install -r requirements.txt

# 3. Test locally
python3 main.py backtest --days 30

# 4. Install as systemd service
sudo tee /etc/systemd/system/crypto-dashboard.service << 'EOF'
[Unit]
Description=Crypto Trading Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/crypto-daytrading
ExecStart=/usr/bin/python3 -m uvicorn web.server:app --host 127.0.0.1 --port 8050
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable crypto-dashboard
sudo systemctl start crypto-dashboard
sudo systemctl status crypto-dashboard
```

## Step 2: Cloudflare Tunnel

```bash
# 1. Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

# 2. Login to Cloudflare
cloudflared tunnel login
# Opens browser → authorize your domain

# 3. Create tunnel
cloudflared tunnel create crypto-dashboard
# Note the tunnel ID (e.g., abc123-def456-...)

# 4. Configure tunnel
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: <TUNNEL_ID>
credentials-file: /home/$USER/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: trading.yourdomain.com
    service: http://localhost:8050
  - service: http_status:404
EOF

# 5. Route DNS
cloudflared tunnel route dns crypto-dashboard trading.yourdomain.com

# 6. Install as service
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl status cloudflared
```

## Step 3: Cloudflare Access (Authentication)

在 Cloudflare Dashboard (https://one.dash.cloudflare.com/) 中:

1. **Access → Applications → Add an application**
2. Type: **Self-hosted**
3. Application name: `Crypto Dashboard`
4. Session duration: `24 hours`
5. Application domain: `trading.yourdomain.com`
6. **Add policy:**
   - Policy name: `Allow me`
   - Action: `Allow`
   - Include: `Emails` → 填入你的邮箱
7. Save

现在访问 `trading.yourdomain.com` 会先要求输入邮箱 → 收到验证码 → 验证后进入 dashboard。

## Step 4: Slack Notifications

### 4.1 Create Slack Webhook

1. 打开 https://api.slack.com/apps → **Create New App** → From scratch
2. App name: `Crypto Trading Bot`, Workspace: 选你的
3. **Incoming Webhooks** → Activate → **Add New Webhook to Workspace**
4. 选择一个 channel (如 `#trading-alerts`)
5. 复制 Webhook URL

### 4.2 Configure on server

```bash
# Set the webhook URL as environment variable
echo 'SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz' >> ~/.bashrc
source ~/.bashrc

# Or add to systemd service:
sudo systemctl edit crypto-dashboard
# Add under [Service]:
# Environment=SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
```

## Step 5: Verify

```bash
# Check dashboard is running
curl http://localhost:8050/api/data | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK, trades:', len(d['timeline']))"

# Check tunnel
curl https://trading.yourdomain.com  # Should redirect to Cloudflare Access login

# Check Slack (test notification)
python3 -c "from web.notify import send_slack; send_slack('Test notification from crypto dashboard')"
```

## Maintenance

```bash
# Update code
cd ~/crypto-daytrading && git pull
sudo systemctl restart crypto-dashboard

# View logs
journalctl -u crypto-dashboard -f

# View tunnel logs
journalctl -u cloudflared -f
```
