#!/bin/bash
# Run immediately before stopping the instance to create an AMI.
#
# cloud-init records in /var/lib/cloud that it has already handled this
# instance. Baking that state into an image can make a new instance skip its
# per-instance modules, so anything driven by user data silently never runs.
# Clearing it is a standard AMI step and easy to forget.
set -euo pipefail

echo "==> stopping services for a clean snapshot"
systemctl stop image2genre nginx || true

echo "==> clearing cloud-init state"
cloud-init clean --logs || true
rm -rf /var/lib/cloud/instances/* || true

echo "==> trimming build-time logs"
journalctl --rotate || true
journalctl --vacuum-time=1s || true

echo "==> clearing shell history"
rm -f /root/.bash_history /home/*/.bash_history || true

echo
echo "Ready. Stop the instance, then EC2 > Actions > Image and templates > Create image."
echo "Leave the launch template's user data EMPTY: associate-eip.service handles the"
echo "Elastic IP on every boot, which user data cannot do."
