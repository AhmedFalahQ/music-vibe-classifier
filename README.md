# 🎵 Music Vibe Classifier — Image2Genre Recommender

Image2Genre predicts the music genre that best matches an image’s visual vibe using a deep learning model trained with Amazon SageMaker. It then recommends curated YouTube playlists (both Global and Khaleeji) based on that genre.
---
It also uses Grad-CAM and a vision model on Amazon Bedrock to provide human-like explanations for its predictions.
---

## 🚀 Features

* 🖼️ Upload an image → get a predicted **music genre**
* 🔥 View a **Grad-CAM heatmap** highlighting areas the model focused on
* 🧠 Get human-readable explanations from a **vision model on Amazon Bedrock** for both the original and heatmapped images
* 🌍 **Bilingual interface (English / Arabic)** with full RTL layout; explanations are generated in the language you are viewing
* 🤖 **ResNet34 model** trained in two phases (frozen & fine-tuned) using **SageMaker**
* 📆 Predictions served via a **Flask app** on **EC2 Spot instance**
* 🎵 Automatically fetches **YouTube playlists** for both global and local (Khaleeji) musical tastes
* ☁️ Uploaded images are sent to **AWS Lambda**, which stores them in **S3** for future retraining
* 🔐 Uses **AWS Secrets Manager** for secure API key and resources variables and config management
* 🪪 Bedrock is reached with the **EC2 instance role** — no model API key is stored anywhere
* ✨ Modern, responsive UI using CSS and `<iframe>` playlist previews

---

## 📂 Project Structure

```
music-vibe-classifier/
├── app.py                  ← Flask app (frontend + inference)
├── model.pth               ← (manually added to EC2)
├── label_encoder.pkl       ← (manually added to EC2)
├── requirements.txt        ← Python dependencies
├── .gitignore              ← Excludes model files, env, cache
├── README.md               ← You're reading it now
├── sagemaker/ 
│   └── train.py            ← Training script 
│   └── sagemaker.py        ← SageMaker script
├── static/
│   └── styles.css          ← Modern UI styling
├── templates/
│   └── index.html          ← Upload + prediction interface
├── utils/
│   ├── __init__.py
│   └── aws_utils.py        ← Lambda invocation to store images in S3
```

---

## 🧐 Model & Training

* Base: **ResNet34**
* Trained in two phases:

  * Phase 1: Only final layer trained
  * Phase 2: Layers 2–4 + FC fine-tuned
* Loss: `CrossEntropyLoss` with class weights
* Metrics: Validation **Accuracy**, **F1 Score**, **Loss Curve**
* Training done on **Amazon SageMaker**
* Model artifact: `model.pth` + `label_encoder.pkl`

### 📊 Performance

* **Final Train Loss:** \~0.25
* **Final Validation Loss:** \~1.2
* **Final Validation Accuracy:** \~70%
* **Epochs:** 15
* **Dataset Size:** \~25,000 High resolution images
* **Split:** 80% training / 20% validation

### 📦 Dataset

* **Source:** Unsplash image dataset (cleaned by project author)
* **Labels:** Manually labeled by the author using AI tools and human curation
* **Genres:** classical, jazz, rock, pop, electronic

* **Preprocessing:**
 * Resized to 224×224
 * Normalized using ImageNet stats
 * Corrupt images filtered using PIL
 * **Label Encoding:** Mapped genre labels to integers via LabelEncoder, saved as label_encoder.pkl

---

## 🧠 Vision-Model Explanations (Amazon Bedrock)
* **Original Image Caption + Justification**
   Explains how the original image visually matches the predicted music genre.
* **Grad-CAM Explanation**
   A second call reasons over the heatmap, highlighting what the model "focused on" and why that supports the genre.
* 🌍 Both are generated in the interface language (English or Arabic).
* 💬 **Responses are structured using prompt engineering.**

### Configuration

Calls go through Bedrock's **Converse API**, which takes the same request shape for
every model on Bedrock — so switching models is a config change, not a code change:

| Variable | Default | Notes |
| ---------------------- | -------------------------- | ------------------------------------------------- |
| `BEDROCK_MODEL_ID`     | `us.amazon.nova-lite-v1:0` | `us.` is the cross-region inference profile; the in-region id is `amazon.nova-lite-v1:0` |
| `BEDROCK_REGION`       | `us-east-1`                | Must be a region where the model is available     |
| `BEDROCK_MAX_TOKENS`   | `200`                      | Explanations are 1–2 sentences                    |

The model must first be enabled under **Bedrock → Model access**, and the instance
role needs `bedrock:InvokeModel` on it. If a call fails the app logs the reason and
renders without that explanation — the genre, heatmap and playlists still appear.

> **Note on Arabic:** Nova Lite is a small, low-cost model. If the Arabic
> explanations read poorly, point `BEDROCK_MODEL_ID` at a Claude model on Bedrock
> instead — no code change required.

---

## ☁️ AWS Services Used

| Service             | Purpose                                        |
| ------------------- | ---------------------------------------------- |
| **SageMaker**       | Model training + model artifact output         |
| **Lambda**          | Stores uploaded images in S3                   |
| **S3**              | Stores images for future retraining            |
| **EC2 (Spot)**      | Hosts the Flask app for cost-efficient serving |
| **Bedrock**         | Vision model for the explanations (instance-role auth) |
| **Secrets Manager** | Secure API key management (e.g., YouTube key)  |
| **NGINX**           | Serves Flask app via reverse proxy             |

---

## 📸 Sample Flow

1. User uploads a photo.
2. Flask app predicts genre using `model.pth`.
3. Grad-CAM heatmap is generated.
4. Two vision-model explanations are returned (original + heatmap), in the interface language.
5. YouTube playlists for Original + Khaleeji are shown.
6. The image is sent to Lambda → stored in S3.
7. The user can listen via embedded YouTube previews.

---

## 📊 Future Ideas

* ⏳ Auto-trigger SageMaker retraining from new S3 images
* 📊 Track prediction analytics / user feedback
* 🌟 Upload feedback to DynamoDB for model tuning

---
## 🤝 Contributors and Contributions

- **Ahmed AlQahtani**  
  - ☁️ Handled all AWS-related development — including EC2 setup, Lambda, S3, and Secrets Manager
  - 🧹 Collected and cleaned the dataset
  - 🧪 Integrated and managed the training pipeline on Amazon SageMaker
  - 🎵 Developed YouTube API features for playlist recommendations
  - 🧱 Built core backend infrastructure and deployment flow

- **Abdulmajeed AlSharafi**  
  - 🔥 Implemented Grad-CAM heatmap visualization
  - 🤖 Integrated GPT-4o explanations (original + Grad-CAM) into the app
  - 🧠 Contributed significantly to **prompt engineering** and tuning GPT outputs
  - 🛠️ Participated in major code reviews and architectural feedback

- **Shared Tasks**  
  - 🎨 Collaborated on UI planning and layout design
  - 🧪 Conducted testing across features and edge cases
  - 📝 Co-authored the documentation and README
---

## 👤 Acknowledgments

> This project was designed and implemented by **Ahmed Alqahtani** and **Abdulmajeed Alsharafi**.
>
> 🧵 *The frontend (Flask + HTML/CSS) and some AWS automation tasks were developed with the help of AI tools as a coding assistants.*
>
> AI tools were instrumental in helping troubleshoot, structure, and speed up development across the UI, backend, and cloud integration. All architectural decisions, testing, and deployment were done independently.

---

## 🎓 Authors

**Ahmed AlQahtani** — [AhmedFalahQ (Ahmed AlQahtani)](https://github.com/AhmedFalahQ)
**Abdulmajeed AlSharafi** — [AbdulmajeedAlsharafi (Abdulmajeed Alsharafi)](https://github.com/AbdulmajeedAlsharafi)
#
