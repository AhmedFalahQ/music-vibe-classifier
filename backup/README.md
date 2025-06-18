# 🎵 Music Vibe Classifier — Image2Genre Recommender

This project predicts a music genre from an image using a deep learning model trained with **Amazon SageMaker**. It then recommends **YouTube playlists** (Original & Khaleeji) that match the image's vibe.
---

## 🚀 Features

* 🖼️ Upload an image → get a predicted **music genre**
* 🤖 **ResNet34 model** trained in two phases (frozen & fine-tuned) using **SageMaker**
* 📆 Predictions served via a **Flask app** on **EC2 Spot instance**
* 🎵 Automatically fetches **YouTube playlists** for both global and local (Khaleeji) musical tastes
* ☁️ Uploaded images are sent to **AWS Lambda**, which stores them in **S3** for future retraining
* 🔐 Uses **AWS Secrets Manager** for secure API key and resources variables and config management
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

## ☁️ AWS Services Used

| Service             | Purpose                                        |
| ------------------- | ---------------------------------------------- |
| **SageMaker**       | Model training + model artifact output         |
| **Lambda**          | Stores uploaded images in S3                   |
| **S3**              | Stores images for future retraining            |
| **EC2 (Spot)**      | Hosts the Flask app for cost-efficient serving |
| **Secrets Manager** | Secure API key management (e.g., YouTube key)  |
| **NGINX**           | Serves Flask app via reverse proxy             |

---

## 📸 Sample Flow

1. User uploads a photo.
2. Flask app predicts genre using `model.pth`.
3. YouTube playlists for Original + Khaleeji are shown.
4. The image is sent to Lambda → stored in S3.
5. The user can listen via embedded YouTube previews.

---

## 📊 Future Ideas

* ⏳ Auto-trigger SageMaker retraining from new S3 images
* 💡 Multilingual interface
* 📊 Track prediction analytics / user feedback
* 🌟 Upload feedback to DynamoDB for model tuning

---

## 👤 Acknowledgments

> This project was designed and implemented by **Ahmed Alqahtani**.
>
> 🧵 *The frontend (Flask + HTML/CSS) and some AWS automation tasks were developed with the help of AI tools as a coding assistants.*
>
> AI tools were instrumental in helping troubleshoot, structure, and speed up development across the UI, backend, and cloud integration. All architectural decisions, testing, and deployment were done independently.

---

## 🎓 Author

**Ahmed AlQahtani** — [AhmedFalahQ (Ahmed AlQahtani)](https://github.com/AhmedFalahQ)
#
