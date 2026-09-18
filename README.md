# 🎵 Image2Genre — Music Vibe Classifier

**Upload a photo, get the music genre that matches its visual vibe — plus the heatmap showing *why*, an explanation in plain language, and playlists to actually listen to.**

Live at **[image2genre.ahmedfalah.dev](https://image2genre.ahmedfalah.dev)** · Bilingual 🇬🇧 English / 🇸🇦 العربية

A ResNet34 trained on Amazon SageMaker does the classification. Grad-CAM shows which
pixels drove the decision. A vision model on Amazon Bedrock reads both the photo and
the heatmap and explains the call in the language you're browsing in. Then it hands
you two YouTube playlists — one global, one Khaleeji.

---

## ✨ What it does

| | |
| --- | --- |
| 🖼️ **Classify** | Photo in → one of five genres out: classical, jazz, rock, pop, electronic |
| 📊 **Show its work** | Confidence distribution across the top 3 genres, not just the winner |
| 🔥 **Grad-CAM heatmap** | The regions the final conv layer actually responded to |
| 🧠 **Explain** | Two Bedrock calls — one reads the photo, one reads the heatmap |
| 🌍 **Bilingual** | Full English / Arabic UI with real RTL layout; explanations generated in the viewing language |
| 🎧 **Recommend** | Global + Khaleeji playlists per genre, previewable inline via `<iframe>` |
| 📱 **Phone-friendly** | HEIC uploads decode correctly; results fit one screen without scrolling |
| 🗂️ **Collect** | Uploads go to Lambda → S3, building a dataset for future retraining |

---

## 🏗️ How a request flows

```
 Browser ──HTTPS──▶ CloudFront ──HTTP──▶ nginx ──▶ gunicorn ──▶ Flask (app.py)
                                                                    │
                            ┌───────────────────────────────────────┤
                            ▼                                       ▼
                   ResNet34 + Grad-CAM                        Lambda ──▶ S3
                            │                                (async archive)
                            ▼
                   Bedrock Converse API ──▶ two explanations (EN or AR)
                            │
                            ▼
                   YouTube Data API ──▶ playlist tracks
```

CloudFront terminates TLS and fronts a single spot instance holding an Elastic IP.
Nothing is cached on the default behaviour — the app is dynamic and the `?lang=`
query string decides the whole page — but `/static/*` is.

---

## 📂 Project structure

```
music-vibe-classifier/
├── app.py                      ← Flask app: inference, Grad-CAM, Bedrock, routing
├── translations.py             ← 46 UI strings × EN/AR, genre labels, RTL direction
├── requirements.txt            ← Serving dependencies (what EC2 installs)
├── requirements-train.txt      ← Training-only: pandas, matplotlib, sagemaker SDK
│
├── model.pth                   ← Not in git — pulled from S3 at build time
├── label_encoder.pkl           ← Not in git — pulled from S3 at build time
│
├── templates/
│   └── index.html              ← Landing, analyzing and results views in one template
├── static/
│   ├── styles.css              ← Dark theme, CSS custom properties, logical properties for RTL
│   └── app.js                  ← Dropzone, analyzing overlay, image/playlist tabs
│
├── utils/
│   ├── images.py               ← Shared decoder: Pillow → HEIF → imageio fallback chain
│   └── aws_utils.py            ← Lambda invocation to archive uploads in S3
│
├── sagemaker/
│   ├── train.py                ← Two-phase training script
│   └── sagemaker.py            ← Job launcher
│
├── deploy/                     ← Reproducible AL2023 build — see deploy/README.md
│   ├── build-ami.sh            ← Idempotent one-shot provisioning
│   ├── verify.sh               ← Proves the box serves the app before you bake
│   ├── prepare-for-ami.sh      ← Clears cloud-init state so the AMI boots clean
│   ├── associate-eip.{sh,service} ← Claims the Elastic IP on every boot
│   ├── image2genre.service     ← gunicorn under systemd
│   ├── nginx-image2genre.conf  ← Reverse proxy + direct /static/ serving
│   └── instance-role-policy.json ← Least-privilege IAM for the instance role
│
└── tests/                      ← No framework, no AWS, no credentials required
    ├── test_gradcam.py         ← Thread-safety and correctness of the heatmap
    ├── test_bedrock.py         ← Converse request shape, via botocore's Stubber
    ├── test_image_loading.py   ← HEIC / truncated / exotic-shape decode paths
    └── test_ui_visibility.py   ← Drives Chromium; asserts what is actually visible
```

---

## 🧠 Model & training

* **Base:** ResNet34, ImageNet-pretrained, with a `Dropout + Linear` head
* **Phase 1:** train the final layer only
* **Phase 2:** unfreeze layers 2–4 and fine-tune
* **Loss:** `CrossEntropyLoss` with class weights
* **Platform:** Amazon SageMaker
* **Artifacts:** `model.pth` + `label_encoder.pkl`

### 📊 Performance

| Metric | Value |
| --- | --- |
| Final train loss | ~0.25 |
| Final validation loss | ~1.2 |
| Validation accuracy | ~70% |
| Epochs | 15 |
| Dataset | ~25,000 high-resolution images |
| Split | 80 / 20 |

### 📦 Dataset

* **Source:** Unsplash, cleaned by the project authors
* **Labels:** manually curated — AI-assisted, human-verified
* **Genres:** classical · jazz · rock · pop · electronic
* **Preprocessing:** resize to 224×224 → ImageNet normalization → corrupt images filtered with PIL
* **Encoding:** genre strings → integers via `LabelEncoder`, persisted as `label_encoder.pkl`

---

## 🔥 Grad-CAM

Hooks on `model.layer4[2].conv2` capture the forward activations and the gradient of
the predicted logit. Channel weights are the spatially averaged gradients; the map is
their ReLU'd weighted sum, upsampled and blended over the original image.

The implementation is **thread-safe**. Capture state is per-call rather than module-level,
hooks are registered and removed around each call, and the math is out-of-place — gunicorn
runs multiple threads per worker, and the earlier version returned heatmaps belonging to
other people's images under concurrency. `tests/test_gradcam.py` reproduces that failure
across 8 threads and asserts it no longer happens.

---

## 💬 Vision explanations on Amazon Bedrock

Two calls per prediction:

1. **Original image** — how the photo's content, colour and mood match the predicted genre
2. **Heatmap** — what the model focused on, and why that supports the call

Both go through the **Converse API**, which takes an identical request shape for every
model on Bedrock. Switching models is a config change, not a code change.

| Variable | Default | Notes |
| --- | --- | --- |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-lite-v1:0` | `us.` prefix is the cross-region inference profile; the in-region id drops it |
| `BEDROCK_REGION` | `us-east-1` | Must be a region where the model is available |
| `BEDROCK_MAX_TOKENS` | `200` | Explanations are 1–2 sentences |

Authentication is the **EC2 instance role** — there is no model API key anywhere in the
repo, the environment, or Secrets Manager. The model must be enabled under
**Bedrock → Model access**, and the role needs `bedrock:InvokeModel` on it.

Explanation failures are non-fatal: the reason is logged and the page renders without
that paragraph. The genre, heatmap, confidence bars and playlists still appear.

> **Note on Arabic:** Nova Lite is a small, low-cost model. If Arabic explanations read
> poorly, point `BEDROCK_MODEL_ID` at a stronger multilingual model on Bedrock — no code
> change required.

---

## 🌍 Bilingual interface

`translations.py` holds all 46 user-facing strings in both languages, plus genre labels,
descriptions, per-genre accent colours and text direction. `?lang=ar` sets `dir="rtl"` on
`<html>` and `data-lang="ar"` on `<body>`; the stylesheet leans on logical properties
(`margin-inline`, `padding-block`) so the layout mirrors without a second stylesheet.

Arabic switches to **IBM Plex Sans Arabic**, with letter-spacing forced to `0` and leading
increased for descenders and diacritics. Arabic is cursive — tracking that flatters Latin
type breaks the joins outright.

---

## 🚀 Running locally

```bash
git clone https://github.com/AhmedFalahQ/music-vibe-classifier.git
cd music-vibe-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place model.pth and label_encoder.pkl in the repo root (gitignored)

export AWS_PROFILE=your-profile        # for Bedrock, Secrets Manager, Lambda
python app.py                          # http://localhost:5000
```

Secret lookups fall back to environment variables (`YOUTUBE_API_KEY`, `bucket_name`,
`lambda_function_name`) when Secrets Manager is unreachable, so the app starts and
classifies without AWS at all. Without credentials you lose the Bedrock explanations, the
S3 archive and the playlist track listings; the genre, heatmap and confidence bars still
work.

### Tests

No pytest, no AWS, no API keys. Each file runs standalone:

```bash
python3 tests/test_gradcam.py
python3 tests/test_bedrock.py
python3 tests/test_image_loading.py
python3 tests/test_ui_visibility.py    # needs playwright + chromium; skips cleanly without
```

---

## ☁️ AWS services

| Service | Purpose |
| --- | --- |
| **SageMaker** | Model training and artifact output |
| **EC2 (Spot)** | Hosts the Flask app — cost-efficient serving behind an ASG |
| **Bedrock** | Vision model for explanations, authenticated by instance role |
| **Lambda** | Archives uploaded images |
| **S3** | Stores model artifacts and collected images for retraining |
| **Secrets Manager** | YouTube API key, bucket name, Lambda function name |
| **CloudFront** | TLS termination, custom domain, edge caching for `/static/*` |
| **Route 53** | `image2genre` → distribution, `image2genre-origin` → Elastic IP |
| **ACM** | Wildcard certificate in `us-east-1` |
| **nginx** | Reverse proxy; serves static assets directly |

Full deployment walkthrough — AMI baking, spot replacement, CloudFront, domain setup and
troubleshooting — lives in **[`deploy/README.md`](deploy/README.md)**.

---

## 📊 Future ideas

* ⏳ Auto-trigger SageMaker retraining from newly collected S3 images
* 📈 Prediction analytics and user feedback capture
* 🗳️ Feedback to DynamoDB for model tuning
* 🎼 Replace hardcoded playlist IDs with live YouTube search
* 📉 Drop the `label_encoder.pkl` pickle for a JSON class list, removing scikit-learn from the serving image

---

## 🤝 Contributors

### **Ahmed AlQahtani** — [@AhmedFalahQ](https://github.com/AhmedFalahQ)
* ☁️ All AWS architecture and development — EC2, Lambda, S3, Secrets Manager, IAM, CloudFront, Route 53
* 🧹 Collected, cleaned and labelled the dataset
* 🧪 Built and ran the SageMaker training pipeline
* 🎵 YouTube API integration and playlist curation, including the Khaleeji selections
* 🧱 Core backend and the entire deployment operation
* 🇸🇦 Wrote and refined the Arabic copy — every translation was rewritten by hand for tone and accuracy

### **Abdulmajeed AlSharafi** — [@AbdulmajeedAlsharafi](https://github.com/AbdulmajeedAlsharafi)
* 🔥 Implemented the Grad-CAM heatmap visualization
* 🤖 Built the original vision-explanation feature (GPT-4o, since migrated to Bedrock)
* 🧠 Prompt engineering and output tuning
* 🛠️ Code review and architectural feedback

### **Claude Code** (Anthropic) — AI pair programmer
Worked directly in the repo across six merged pull requests:
* 🎨 Rebuilt the frontend — dark theme, single-screen results layout, and the full bilingual EN/AR system with RTL (`index.html`, `styles.css`, `app.js`, `translations.py` scaffolding)
* 🧵 Found and fixed a **Grad-CAM concurrency bug** that served wrong heatmaps under load — reproduced at 8 threads, then fixed with per-call capture and out-of-place math
* 🎨 Found and fixed **inverted heatmap colours** — OpenCV returns BGR, Pillow was reading it as RGB, so every explanation had been describing the *coldest* region of the image
* ☁️ Migrated vision explanations from OpenAI to the **Bedrock Converse API**, eliminating the stored API key in favour of instance-role auth
* 📱 Fixed **HEIC decoding** so iPhone uploads work, and replaced `print` with structured logging and tracebacks
* 🔧 Repaired `requirements.txt` (UTF-16 encoded and unparseable by pip, with conflicting OpenCV builds)
* 🚢 Wrote the **`deploy/` toolchain** — AL2023 provisioning, AMI bake preparation, boot-time Elastic IP association, nginx and systemd units, IAM policy, and a verification script that fails loudly
* 🧪 Wrote the **`tests/` suite**, each test anchored to a bug that actually reached production

---

## 👤 Acknowledgments

> Image2Genre was designed and built by **Ahmed AlQahtani** and **Abdulmajeed AlSharafi**.
>
> 🧵 A significant share of the frontend, the deployment tooling and the test suite was
> written by **Claude Code**, working in the repository as a collaborator rather than as
> autocomplete — including several production bug fixes credited above. Every change was
> reviewed, tested and merged by the authors, who own all architectural decisions, the
> model, the dataset and the infrastructure.

---

## 🎓 Authors

**Ahmed AlQahtani** — [@AhmedFalahQ](https://github.com/AhmedFalahQ)
**Abdulmajeed AlSharafi** — [@AbdulmajeedAlsharafi](https://github.com/AbdulmajeedAlsharafi)
