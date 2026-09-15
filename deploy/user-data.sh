#!/bin/bash
# EC2 user data for music-vibe-classifier on a spot instance.
#
# Runs on every boot, so it must be idempotent: a replaced spot instance
# re-runs this from scratch and has to come back serving on the same address.
#
# Assumes Ubuntu 24.04 LTS. Bake an AMI after the first successful run -- the
# torch install is the slow part and this script skips it when the venv exists.
set -euxo pipefail
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

# ---------------------------------------------------------------- settings --
REPO_URL="https://github.com/AhmedFalahQ/music-vibe-classifier.git"
BRANCH="main"
APP_DIR="/opt/image2genre"
APP_USER="appuser"

EIP_ALLOC_ID="eipalloc-REPLACE_ME"        # the Elastic IP Route 53 points at
MODEL_S3="s3://REPLACE_ME/models"         # holds model.pth + label_encoder.pkl
AWS_REGION_DEFAULT="us-east-1"

# --------------------------------------------------------------- metadata ---
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
meta() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
INSTANCE_ID=$(meta instance-id)
REGION=$(meta placement/region || echo "$AWS_REGION_DEFAULT")

# ------------------------------------------ claim the stable public address --
# Route 53 points at this EIP permanently, so a replaced instance takes over
# the same address with no DNS change and no propagation delay.
aws ec2 associate-address --region "$REGION" \
  --instance-id "$INSTANCE_ID" \
  --allocation-id "$EIP_ALLOC_ID" \
  --allow-reassociation

# ------------------------------------------------------- packages (once) ----
if [ ! -x /usr/bin/nginx ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y python3-venv python3-pip git nginx awscli
fi

id -u "$APP_USER" &>/dev/null || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"

# ------------------------------------------------------------------- code ---
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

# --------------------------------------------------- model artifacts (S3) ---
# Gitignored, so they are never in the clone. Without them the app cannot boot.
aws s3 cp "$MODEL_S3/model.pth"          "$APP_DIR/model.pth"          --region "$REGION"
aws s3 cp "$MODEL_S3/label_encoder.pkl"  "$APP_DIR/label_encoder.pkl"  --region "$REGION"

# ----------------------------------------------------- python environment ---
# The expensive step. Present already on a baked AMI, so this is a no-op then.
if [ ! -d "$APP_DIR/.venv" ]; then
  python3 -m venv "$APP_DIR/.venv"
  "$APP_DIR/.venv/bin/pip" install --upgrade pip
  "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
  "$APP_DIR/.venv/bin/pip" install gunicorn
fi

chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# ------------------------------------------------------------- gunicorn -----
# 2 workers: each loads its own copy of torch + the model (~1.5GB), so this
# needs a t3.large. On a t3.medium use -w 1 --threads 8 instead.
# Timeout 120 because a request runs inference plus two Bedrock calls.
cat > /etc/systemd/system/image2genre.service <<UNIT
[Unit]
Description=Image2Genre
After=network-online.target
Wants=network-online.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment="BEDROCK_REGION=$REGION"
ExecStart=$APP_DIR/.venv/bin/gunicorn --workers 2 --threads 4 --timeout 120 \\
          --bind 127.0.0.1:5000 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# ---------------------------------------------------------------- nginx -----
cat > /etc/nginx/sites-available/image2genre <<'NGINX'
server {
    listen 80 default_server;
    server_name _;

    # Phone photos are large, and the upload is multipart.
    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location /static/ {
        alias /opt/image2genre/static/;
        expires 7d;
        add_header Cache-Control "public";
    }
}
NGINX
ln -sf /etc/nginx/sites-available/image2genre /etc/nginx/sites-enabled/image2genre
rm -f /etc/nginx/sites-enabled/default
nginx -t

systemctl daemon-reload
systemctl enable --now image2genre
systemctl restart nginx
