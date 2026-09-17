# Deploying Image2Genre

Amazon Linux 2023, a baked AMI, and a spot instance behind CloudFront.

Everything the instance needs is in the AMI. The only per-boot work is claiming
the Elastic IP, which `associate-eip.service` does — **not** user data, which
runs once per *instance* and so never re-runs on a stop/start.

| File | Purpose |
| --- | --- |
| `build-ami.sh` | One-shot build of a fresh AL2023 instance. Idempotent. |
| `verify.sh` | Proves the box is serving the app. Run before baking. |
| `prepare-for-ami.sh` | Clears cloud-init state and logs. Run before stopping. |
| `image2genre.service` | gunicorn under systemd |
| `associate-eip.service` / `associate-eip.sh` | Claims the Elastic IP every boot |
| `nginx-image2genre.conf` | Reverse proxy; serves `/static/` directly |
| `instance-role-policy.json` | Least-privilege IAM policy for the instance role |

## Prerequisites

1. **ACM certificate in `us-east-1`** covering the domain. CloudFront accepts no
   other region. A `*.example.dev` wildcard covers one level of subdomain.
2. **Bedrock model access** for Amazon Nova Lite in `us-east-1`
   (Bedrock console → Model access). Without it explanations come back empty.
3. **Secrets in `us-east-1`**: `youtube/api_key` (`{"api_key": "..."}`) and
   `app/keys` (`{"bucket_name": "...", "lambda_function_name": "..."}`).
   Missing secrets are swallowed at startup and show up as empty playlists.
4. **Model artifacts in S3** under `s3://BUCKET/models/`:
   ```
   aws s3 cp model.pth         s3://BUCKET/models/
   aws s3 cp label_encoder.pkl s3://BUCKET/models/
   ```
   The checkpoint must be **ResNet34 with a `Dropout + Linear` head**. Verify
   before uploading:
   ```bash
   python - <<'PY'
   import torch
   sd = torch.load("model.pth", map_location="cpu"); sd = sd.get("state_dict", sd)
   blocks = {}
   for k in sd:
       if k.startswith("layer"):
           L, B = k.split(".")[0], int(k.split(".")[1])
           blocks[L] = max(blocks.get(L, 0), B + 1)
   print("layout:", [blocks.get(f"layer{i}") for i in (1,2,3,4)], "want [3, 4, 6, 3]")
   print("head  :", [k for k in sd if k.startswith("fc")], "want fc.1.weight / fc.1.bias")
   PY
   ```
   `[2, 2, 2, 2]` means ResNet18 — the wrong file.
5. **IAM role** `image2genre-role` with `instance-role-policy.json` (fill in the
   `REPLACE_ME` values) plus `AmazonSSMManagedInstanceCore`.
6. **Elastic IP** allocated. Note the allocation id (`eipalloc-…`).

## Build

Launch **Amazon Linux 2023**, t3.large, 30 GB gp3, IAM profile
`image2genre-role`, security group allowing **80 from anywhere** for now. No key
pair — connect with Session Manager.

```bash
sudo dnf install -y git
git clone https://github.com/AhmedFalahQ/music-vibe-classifier.git /tmp/i2g
sudo MODEL_BUCKET=YOUR_BUCKET EIP_ALLOCATION_ID=eipalloc-YOURS \
     bash /tmp/i2g/deploy/build-ami.sh
sudo bash /tmp/i2g/deploy/verify.sh
```

`verify.sh` must print `ALL CHECKS PASSED`. Then **upload a real HEIC through a
browser** at the instance's public IP and confirm genre, heatmap, confidence
bars, explanations *and* playlists. Empty playlists mean the secrets are
missing; empty explanations mean Bedrock access is.

## Bake

```bash
sudo bash /tmp/i2g/deploy/prepare-for-ami.sh
```

Stop the instance, then **Actions → Image and templates → Create image**.

## Launch template and ASG

Launch template: the new AMI, t3.large, `image2genre-sg`, `image2genre-role`,
and **user data empty**.

Auto Scaling group: 100% Spot, capacity-optimized, min/desired/max = 1, with
`t3.large`, `t3a.large` and `m5.large` listed so it can find capacity.

Verify replacement works before going further: terminate the instance and
confirm `curl http://ELASTIC_IP/health` recovers within a couple of minutes.

## Domain

1. **Origin record** — in the existing hosted zone, `A` record
   `image2genre-origin` → the Elastic IP, TTL 300.
2. **CloudFront** — origin `image2genre-origin.example.dev` (type it; it is not
   in the dropdown), HTTP only on port 80, **origin response timeout 120**.
   Default behaviour: redirect HTTP to HTTPS, **allow POST** (uploads fail
   without it), cache policy **CachingDisabled** (the standard policy strips
   query strings, which would serve cached English to everyone and make
   `?lang=ar` look broken), origin request policy AllViewerExceptHostHeader.
   Alternate domain name `image2genre.example.dev`, and the us-east-1
   certificate. Afterwards add a `/static/*` behaviour with CachingOptimized.
3. **Site record** — `A` alias `image2genre` → the CloudFront distribution.
4. **Lock the origin** once HTTPS works: replace the `0.0.0.0/0` rule on port 80
   with the prefix list `com.amazonaws.global.cloudfront.origin-facing`.

### `.dev` domains

`.dev` is HSTS-preloaded, so browsers force HTTPS on every hostname under it and
refuse plain HTTP outright. Test the origin with `curl`, never a browser — a
browser hitting `http://image2genre-origin.example.dev` gets upgraded to port
443, where nothing is listening, and reports a timeout that looks like a server
fault.

## Updating a running instance

```bash
cd /opt/image2genre
sudo git pull origin main
sudo -u appuser .venv/bin/pip install -r requirements.txt   # if deps changed
sudo systemctl restart image2genre
journalctl -u image2genre -f -o cat
```

Re-bake the AMI afterwards, or the change is lost on the next spot replacement.

## Watching logs

```bash
journalctl -u image2genre -f -o cat
journalctl -u associate-eip -n 20 --no-pager
sudo tail -f /var/log/nginx/access.log /var/log/nginx/error.log
```

The app logs through `logging`, so failures carry a full traceback.
