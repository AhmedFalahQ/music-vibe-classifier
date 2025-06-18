import boto3
import base64
import os
import io
import requests
import random
from dotenv import load_dotenv
import json
from torchvision import models, transforms
import torch
import joblib
import imageio.v3 as iio
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, render_template
from utils.aws_utils import invoke_lambda_to_store_image

app = Flask(__name__)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)    # Load Enviroment variables from .env file

# AWS Secrets Manager Client
secrets_client = boto3.client('secretsmanager', region_name='us-east-1')

def get_secret(secret_name):
    """Retrieve secret from AWS Secrets Manager"""
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except Exception as e:
        print(f"Error fetching secret: {str(e)}")
        raise

# Load keys from AWS Secrets Manager
try:
    YOUTUBE_API_KEY = dict.get(json.loads(get_secret("youtube/api_key")),"api_key")
    secrets=json.loads(get_secret("app/keys"))
    bucket_name=secrets.get("bucket_name")
    lambda_function_name=secrets.get("lambda_function_name")
except:
    # Fallback for local development
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    bucket_name=os.getenv("bucket_name")
    lambda_function_name=os.getenv("lambda_function_name")

PLAYLIST_MAP = { # a map for playlist IDs of original and khaleeji songs for each genre
    "classical": {
        "original": "RDCLAK5uy_mv1P2oVguxLCIDXavV-jcDG1lQyukfSpo",  
        "khaleeji": "PL5B2S7bvHBUagCKY3JC0A-Ww8i4ftZdfz"  
    },
    "pop": {
        "original": "RDCLAK5uy_lb6CVU6S4uVugLVNTU9WhqfaomWAgnho4",   
        "khaleeji": "RDCLAK5uy_ncf7GgjSEDa0f_EiMC2D0px8NZ5RfPuz0"    
    },
    "electronic":{
        "original":"RDCLAK5uy_nIQ-vZjrOsAAHK2SNZZO7mJ0e1yak6baE",
        "khaleeji":"RDCLAK5uy_l23ciejFJWrpUgto4eqmxNLanDwVsbMOI"
    },
    "jazz":{
        "original":"RDCLAK5uy_l3Iir9QZmCUPLjUqyR9PmYIKkkY7YEJZA",
        "khaleeji":"RDCLAK5uy_l1IZ_kCryTVQ63XSwf9kQKSyUlJmWD3HA"
    },
    "rock":{
        "original":"RDCLAK5uy_lvHI2Z7dSfpD5g8wvmePjWPfYwq5IgkLo",
        "khaleeji":"RDCLAK5uy_mfmwWZkgEnBqMcEbBFjmTdIKsuBvW0G5c"

    }
}

GENRE_DESCRIPTIONS = { # a desciption for images that mapped for each genre
    "classical": {
        "text": "calm and inspiring, often associated with cool colors and timeless beauty"
    },
    "pop": {
        "text": "energetic and vibrant, often associated with bright colors, catchy melodies, and modern trends"
    },
    "electronic": {
        "text": "dynamic and futuristic, often associated with pulsating rhythms, abstract shapes, and synthetic sounds"
    },
    "jazz": {
        "text": "sophisticated and smooth, often associated with warm tones, improvisation, and a relaxed atmosphere"
    },
    "rock": {
        "text": "powerful and raw, often associated with bold textures, rebellious energy, and strong emotions"
    }
}

def get_playlist_tracks(playlist_id, api_key, max_results=50):
    """Fetch tracks from a YouTube playlist"""

    url = "https://www.googleapis.com/youtube/v3/playlistItems" # From Youtube API Doc.
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
                "type": "original" if "original" in playlist_id else "khaleeji"  # For UI differentiation
            }
            for item in response.json().get("items", [])
        ]
    except Exception as e:
        print(f"Error fetching playlist {playlist_id}: {str(e)}")
        return []

# Model Loading
model = models.resnet34(pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 5)
)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()
label_encoder = joblib.load("label_encoder.pkl")

# Image preprocessing 
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def load_image(image_bytes):
    try:
        # Try with Pillow first
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        try:
            # Try using imageio supports HEIC with pyav (Mobile pictures)
            img_array = iio.imread(image_bytes)
            return Image.fromarray(img_array).convert("RGB")
        except Exception as err:
            print(f"HEIC fallback failed: {err}")
            raise ValueError("Unsupported or corrupt image format.")

# Prediction function 
def predict(image_bytes):
    image = load_image(image_bytes)
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted_idx])[0]

@app.route("/", methods=["GET", "POST"])
def index():
    image_data = None
    prediction_text = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            try:
                # Read and encode image for display
                image_bytes = file.read()
                image_data = base64.b64encode(image_bytes).decode('utf-8')

                # Call the lambda function invocation
                invoke_lambda_to_store_image(image_bytes,bucket_name,lambda_function_name,region="us-east-1")
                
                prediction = predict(image_bytes)
                prediction_info = GENRE_DESCRIPTIONS.get(prediction, {
                    "text": f"an intriguing mix, reminiscent of the {prediction} genre."
                })

                prediction_text = prediction_info["text"]
                # Fetch ALL available tracks from the original playlist
                all_original_tracks = get_playlist_tracks(
                    PLAYLIST_MAP.get(prediction, {}).get("original", ""), 
                    YOUTUBE_API_KEY
                )
                
                # Random sample from the fetched tracks
                num_original_samples = min(10, len(all_original_tracks))
                original_tracks = random.sample(all_original_tracks, num_original_samples) if all_original_tracks else []
                
                # Fetch ALL available tracks from the khaleeji playlist
                all_khaleeji_tracks = get_playlist_tracks(
                    PLAYLIST_MAP.get(prediction, {}).get("khaleeji", ""), 
                    YOUTUBE_API_KEY
                )
                
                # Random sample from the fetched tracks
                num_khaleeji_samples = min(10, len(all_khaleeji_tracks))
                khaleeji_tracks = random.sample(all_khaleeji_tracks, num_khaleeji_samples) if all_khaleeji_tracks else []
                
                return render_template(
                    "index.html",
                    prediction=prediction,
                    prediction_text=prediction_text,
                    original_tracks=original_tracks,
                    khaleeji_tracks=khaleeji_tracks,
                    image_data=image_data,
                    error=None if (original_tracks or khaleeji_tracks) else "No playlists found"
                )
                
            except Exception as e:
                return render_template(
                    "index.html",
                    error=f"Error processing image: {str(e)}",
                    image_data=image_data
                )
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)