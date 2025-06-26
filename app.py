import boto3
import base64
import os
import io
import requests
import random
import json
from dotenv import load_dotenv
from torchvision import models, transforms
import torch
import torch.nn.functional as F
import joblib
import imageio.v3 as iio
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from flask import Flask, request, render_template
from utils.aws_utils import invoke_lambda_to_store_image
from openai import OpenAI

app = Flask(__name__)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# AWS setup
secrets_client = boto3.client('secretsmanager', region_name='us-east-1')

def get_secret(secret_name):
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except Exception as e:
        print(f"Error fetching secret: {str(e)}")
        raise

# Use local .env fallback
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
bucket_name = os.getenv("bucket_name")
lambda_function_name = os.getenv("lambda_function_name")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
PLAYLIST_MAP = {
    "classical": {"original": "RDCLAK5uy_mv1P2oVguxLCIDXavV-jcDG1lQyukfSpo", "khaleeji": "PL5B2S7bvHBUagCKY3JC0A-Ww8i4ftZdfz"},
    "pop": {"original": "RDCLAK5uy_lb6CVU6S4uVugLVNTU9WhqfaomWAgnho4", "khaleeji": "RDCLAK5uy_ncf7GgjSEDa0f_EiMC2D0px8NZ5RfPuz0"},
    "electronic": {"original": "RDCLAK5uy_nIQ-vZjrOsAAHK2SNZZO7mJ0e1yak6baE", "khaleeji": "RDCLAK5uy_l23ciejFJWrpUgto4eqmxNLanDwVsbMOI"},
    "jazz": {"original": "RDCLAK5uy_l3Iir9QZmCUPLjUqyR9PmYIKkkY7YEJZA", "khaleeji": "RDCLAK5uy_l1IZ_kCryTVQ63XSwf9kQKSyUlJmWD3HA"},
    "rock": {"original": "RDCLAK5uy_lvHI2Z7dSfpD5g8wvmePjWPfYwq5IgkLo", "khaleeji": "RDCLAK5uy_mfmwWZkgEnBqMcEbBFjmTdIKsuBvW0G5c"}
}

model = models.resnet34(pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 5)
)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

label_encoder = joblib.load("label_encoder.pkl")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

gradients = None
activations = None

def save_grad(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def save_act(module, input, output):
    global activations
    activations = output

target_layer = model.layer4[2].conv2
model.layer4[2].conv2.register_forward_hook(save_act)
model.layer4[2].conv2.register_backward_hook(save_grad)

def load_image(image_bytes):
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        try:
            img_array = iio.imread(image_bytes)
            return Image.fromarray(img_array).convert("RGB")
        except Exception as err:
            print(f"HEIC fallback failed: {err}")
            raise ValueError("Unsupported or corrupt image format.")

def generate_heatmap(input_tensor, class_idx):
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, class_idx]
    score.backward()

    pooled_grad = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grad[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    return heatmap.detach().numpy()

def overlay_heatmap(heatmap, img, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(img)
    overlay = heatmap_color * alpha + img
    return np.uint8(overlay)

def merge_images_side_by_side(img1, img2):
    merged = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (img1.width, 0))
    return merged

def get_gpt4_explanation(image: Image.Image, label: str) -> str:
    prompt = (
        f"VERY Briefly explain why this image fits the music genre '{label}', "
        "based on the visuals and heatmap highlights. Focus on mood, style, or key elements."
    )
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

def predict(image_bytes):
    image = load_image(image_bytes)
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

    heatmap = generate_heatmap(input_tensor, predicted_idx)
    cam_image = overlay_heatmap(heatmap, image)
    cam_pil = Image.fromarray(cam_image)
    buf = io.BytesIO()
    cam_pil.save(buf, format='JPEG')
    gradcam_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Merge for GPT explanation
    merged = merge_images_side_by_side(image, cam_pil)
    explanation = get_gpt4_explanation(merged, predicted_label)

    return predicted_label, gradcam_base64, explanation

def get_playlist_tracks(playlist_id, api_key, max_results=50):
    url = "https://www.googleapis.com/youtube/v3/playlistItems"
    params = {
        "key": api_key,
        "part": "snippet",
        "playlistId": playlist_id,
        "maxResults": max_results
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return [
            {
                "title": item["snippet"]["title"],
                "artist": item["snippet"].get("videoOwnerChannelTitle", "Unknown Artist"),
                "url": f"https://www.youtube.com/watch?v={item['snippet']['resourceId']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                "type": "original" if "original" in playlist_id else "khaleeji"
            }
            for item in response.json().get("items", [])
        ]
    except Exception as e:
        print(f"Error fetching playlist {playlist_id}: {str(e)}")
        return []

@app.route("/", methods=["GET", "POST"])
def index():
    image_data = None
    gradcam_data = None
    prediction_text = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            try:
                image_bytes = file.read()
                image_data = base64.b64encode(image_bytes).decode('utf-8')
                invoke_lambda_to_store_image(image_bytes, bucket_name, lambda_function_name, region="us-east-1")

                prediction, gradcam_data, prediction_text = predict(image_bytes)

                all_original_tracks = get_playlist_tracks(PLAYLIST_MAP.get(prediction, {}).get("original", ""), YOUTUBE_API_KEY)
                original_tracks = random.sample(all_original_tracks, min(10, len(all_original_tracks))) if all_original_tracks else []

                all_khaleeji_tracks = get_playlist_tracks(PLAYLIST_MAP.get(prediction, {}).get("khaleeji", ""), YOUTUBE_API_KEY)
                khaleeji_tracks = random.sample(all_khaleeji_tracks, min(10, len(all_khaleeji_tracks))) if all_khaleeji_tracks else []

                return render_template(
                    "index.html",
                    prediction=prediction,
                    prediction_text=prediction_text,
                    original_tracks=original_tracks,
                    khaleeji_tracks=khaleeji_tracks,
                    image_data=image_data,
                    gradcam_data=gradcam_data,
                    error=None if (original_tracks or khaleeji_tracks) else "No playlists found"
                )
            except Exception as e:
                return render_template("index.html", error=f"Error processing image: {str(e)}", image_data=image_data)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
