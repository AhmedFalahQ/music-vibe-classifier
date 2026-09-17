#!/bin/bash
# Check the instance is genuinely serving the app before baking an AMI from it.
# Every check distinguishes the app from nginx's default page: a 200 alone does
# not, because nginx's own welcome page also returns 200.
set -uo pipefail
APP_DIR="${APP_DIR:-/opt/image2genre}"
fails=0
ok()   { echo "  PASS  $1"; }
bad()  { echo "  FAIL  $1${2:+  -- $2}"; fails=$((fails+1)); }

echo "services"
for unit in image2genre nginx; do
    systemctl is-active --quiet "$unit" && ok "$unit active" || bad "$unit active"
    systemctl is-enabled --quiet "$unit" && ok "$unit enabled at boot" || bad "$unit enabled at boot"
done
systemctl is-enabled --quiet associate-eip && ok "associate-eip enabled at boot" \
    || bad "associate-eip enabled at boot"

echo
echo "http"
[ "$(curl -s localhost/health)" = "ok" ] && ok "/health" || bad "/health" "$(curl -s localhost/health | head -c 60)"
title=$(curl -s localhost/ | grep -o '<title>[^<]*</title>' || true)
case "$title" in
    *Image2Genre*) ok "/ served by the app  ($title)" ;;
    *)             bad "/ served by the app" "got: ${title:-nothing} -- nginx default block is probably winning" ;;
esac
[ "$(curl -s 'localhost/?lang=ar' | grep -c 'dir="rtl"')" -ge 1 ] && ok "?lang=ar switches to RTL" || bad "?lang=ar switches to RTL"
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost/static/styles.css)" = "200" ] \
    && ok "/static/ served" || bad "/static/ served" "page would render unstyled"

echo
echo "artifacts"
[ -f "$APP_DIR/model.pth" ] && ok "model.pth present" || bad "model.pth present"
[ -f "$APP_DIR/label_encoder.pkl" ] && ok "label_encoder.pkl present" || bad "label_encoder.pkl present"
classes=$(sudo -u "${APP_USER:-appuser}" "$APP_DIR/.venv/bin/python" -c \
    "import joblib;print(sorted(joblib.load('$APP_DIR/label_encoder.pkl').classes_))" 2>/dev/null || echo ERROR)
expected="['classical', 'electronic', 'jazz', 'pop', 'rock']"
[ "$classes" = "$expected" ] && ok "label encoder classes match the app" \
    || bad "label encoder classes match the app" "got $classes -- predictions would be mislabelled"

echo
echo "elastic ip"
journalctl -u associate-eip -n 20 --no-pager 2>/dev/null | grep -q "Associated " \
    && ok "EIP associated" || bad "EIP associated" "see: journalctl -u associate-eip"

echo
if [ "$fails" -eq 0 ]; then
    echo "ALL CHECKS PASSED - upload a real HEIC in a browser, then run prepare-for-ami.sh"
else
    echo "$fails CHECK(S) FAILED - do not bake this image yet"
fi
exit $(( fails > 0 ))
