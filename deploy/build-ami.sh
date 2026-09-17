#!/bin/bash
# Build the Image2Genre AMI on Amazon Linux 2023.
#
#   sudo MODEL_BUCKET=my-bucket EIP_ALLOCATION_ID=eipalloc-0123 \
#        bash deploy/build-ami.sh
#
# Idempotent: safe to re-run on the same instance after a code change.
# Nothing here needs hand-editing, so a typo in a config value cannot end up
# baked into the image.
#
# When it finishes, run deploy/verify.sh, then deploy/prepare-for-ami.sh
# immediately before stopping the instance to create the image.
set -euo pipefail

: "${MODEL_BUCKET:?set MODEL_BUCKET to the bucket holding model.pth and label_encoder.pkl}"
: "${EIP_ALLOCATION_ID:?set EIP_ALLOCATION_ID to your Elastic IP allocation id (eipalloc-...)}"

REPO_URL="${REPO_URL:-https://github.com/AhmedFalahQ/music-vibe-classifier.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-/opt/image2genre}"
APP_USER="${APP_USER:-appuser}"
MODEL_PREFIX="${MODEL_PREFIX:-models}"
AWS_REGION="${AWS_REGION:-us-east-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> packages"
# No awscli here: Amazon Linux 2023 ships CLI v2 already. Installing it from a
# package manager gets you v1, which is what broke the earlier Ubuntu build.
dnf install -y git nginx python3.11 python3.11-pip >/dev/null

echo "==> service account"
id -u "$APP_USER" &>/dev/null || useradd --system --create-home --shell /sbin/nologin "$APP_USER"

echo "==> code"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$APP_DIR" reset --hard "origin/$BRANCH"
else
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

echo "==> model artifacts"
# Gitignored, so never present in the clone. Without them the app refuses to
# start, by design, with a message naming the missing file.
aws s3 cp "s3://$MODEL_BUCKET/$MODEL_PREFIX/model.pth"         "$APP_DIR/model.pth"         --region "$AWS_REGION"
aws s3 cp "s3://$MODEL_BUCKET/$MODEL_PREFIX/label_encoder.pkl" "$APP_DIR/label_encoder.pkl" --region "$AWS_REGION"

echo "==> python environment"
if [ ! -d "$APP_DIR/.venv" ]; then
    python3.11 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
"$APP_DIR/.venv/bin/pip" install --quiet gunicorn
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "==> configuration"
cat > /etc/image2genre.env <<ENVFILE
# Written by build-ami.sh. Read by associate-eip.sh on every boot.
EIP_ALLOCATION_ID=$EIP_ALLOCATION_ID
ENVFILE
chmod 0644 /etc/image2genre.env

echo "==> services"
install -m 0755 "$HERE/associate-eip.sh"       /usr/local/bin/associate-eip.sh
install -m 0644 "$HERE/associate-eip.service"  /etc/systemd/system/associate-eip.service
install -m 0644 "$HERE/image2genre.service"    /etc/systemd/system/image2genre.service
install -m 0644 "$HERE/nginx-image2genre.conf" /etc/nginx/conf.d/image2genre.conf

nginx -t
systemctl daemon-reload
systemctl enable  associate-eip.service
systemctl enable  image2genre.service
systemctl enable  nginx
systemctl restart image2genre
systemctl restart nginx
systemctl start   associate-eip.service || echo "WARNING: EIP association failed; check: journalctl -u associate-eip"

echo
echo "Build complete. Next: sudo bash $HERE/verify.sh"
