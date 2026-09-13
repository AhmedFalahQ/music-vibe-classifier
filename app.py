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
import translations as tr
import numpy as np
import cv2
import threading

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

# Bedrock vision model, reached with the EC2 instance role -- no API key is
# stored or fetched for it. The Converse API takes the same request shape for
# every model on Bedrock, so moving to a different one (a Claude model, if the
# Arabic explanations need more nuance than Nova Lite gives) is a config change
# rather than a code change.
#
# "us." is the cross-region inference profile most accounts need; the in-region
# id is "amazon.nova-lite-v1:0". Either way the model has to be enabled first
# under Bedrock -> Model access.
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "200"))

bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

PLAYLIST_MAP = {        # a map for playlist IDs of original and khaleeji songs for each genre
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
        # Without a timeout a stalled connection hangs the whole request,
        # which looks exactly like the app freezing on the analyzing screen.
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return [
            {
                "title": item["snippet"]["title"],
                "artist": item["snippet"].get("videoOwnerChannelTitle", "Unknown Artist"),
                "url": f"https://www.youtube.com/watch?v={item['snippet']['resourceId']['videoId']}",
                "video_id": item['snippet']['resourceId']['videoId'],
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

TARGET_LAYER = model.layer4[2].conv2  # The last conv layer in resnet34

# Grad-CAM needs an activation and its gradient from the same pass. Flask serves
# requests on threads and they all share this one model, so both the capture and
# the backward pass have to be serialised: backward() also accumulates into the
# model's parameter .grad buffers, which two threads cannot do at once.
_cam_lock = threading.Lock()

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

def generate_heatmap(input_tensor, class_idx):
    """Grad-CAM over the last conv layer, as an H x W float array in [0, 1]."""
    captured = {}

    def save_act(_module, _inputs, output):
        captured["activations"] = output.detach()

    def save_grad(_module, _grad_input, grad_output):
        captured["gradients"] = grad_output[0].detach()

    with _cam_lock:
        forward_handle = TARGET_LAYER.register_forward_hook(save_act)
        # register_full_backward_hook, not the deprecated register_backward_hook,
        # which is documented to report wrong gradients for some modules.
        backward_handle = TARGET_LAYER.register_full_backward_hook(save_grad)
        try:
            model.zero_grad(set_to_none=True)
            output = model(input_tensor)          # Forward pass
            output[0, class_idx].backward()       # Backward pass
        finally:
            # Always unhook: a leaked hook would fire on every later request.
            forward_handle.remove()
            backward_handle.remove()
            model.zero_grad(set_to_none=True)

    activations = captured.get("activations")
    gradients = captured.get("gradients")
    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM captured nothing from the target layer")

    # Channel weights are the spatially averaged gradients. Combined
    # out-of-place: the old code multiplied into the captured activation
    # tensor itself, which mutated a tensor autograd still referenced.
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activations).sum(dim=1)).squeeze(0)

    peak = torch.max(heatmap)
    if peak <= 0:
        # Nothing in the image argued for this class; dividing here gave NaN.
        return np.zeros(tuple(heatmap.shape), dtype=np.float32)
    return (heatmap / peak).numpy()

def overlay_heatmap(heatmap, img, alpha=0.5):
    """Blend a Grad-CAM heatmap over the image. Returns an RGB array."""
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)

    # applyColorMap returns BGR, and this array is handed straight to PIL as
    # RGB. Without the conversion JET comes out inverted: the hottest regions
    # render blue and the coldest red, which is backwards from the legend in the
    # UI and from what the explanation prompt tells the model to look for.
    heatmap_color = cv2.cvtColor(
        cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    img = np.array(img)
    # Weighted blend. Adding the colormap on top of a full-strength image (the
    # previous form) pushed bright areas to white and lost the photo underneath.
    overlay = alpha * heatmap_color + (1 - alpha) * img
    return np.uint8(np.clip(overlay, 0, 255))

def merge_images_side_by_side(img1, img2):
    merged = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (img1.width, 0))
    return merged

def compress_image(image, max_size=(512, 512)):
    image = image.convert("RGB")
    image.thumbnail(max_size)
    return image

def get_vision_explanation(image, genre, lang="en", grad=False):
    """Ask the Bedrock vision model to explain a prediction.

    Returns None instead of raising: losing the caption should not cost the
    user their genre, heatmap and playlists.
    """
    if grad:
        prompt = (
            "Look at this merged image of an original photo and its Grad-CAM heatmap. "
            f"The model predicted the '{genre}' music genre. Based on which parts of the "
            "image are most activated (red-hot in the heatmap), explain in 1-2 sentences "
            f"why the model might associate this with '{genre}'."
        )
    else:
        prompt = (
            "Describe this image in one sentence, then explain in one sentence why it "
            f"fits the music genre '{genre}'."
        )

    # Answer in whichever language the page is being viewed in.
    prompt = f"{prompt} {tr.PROMPT_LANGUAGE.get(lang, tr.PROMPT_LANGUAGE['en'])}"

    buf = io.BytesIO()
    image.save(buf, format="JPEG")

    try:
        response = bedrock_runtime.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    # Converse takes raw bytes -- no base64 wrapper needed.
                    {"image": {"format": "jpeg", "source": {"bytes": buf.getvalue()}}},
                    {"text": prompt},
                ],
            }],
            inferenceConfig={"maxTokens": BEDROCK_MAX_TOKENS, "temperature": 0.3},
        )
        return response["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        # AccessDeniedException here almost always means this model id has not
        # been enabled under Bedrock -> Model access for this account/region.
        print(f"Bedrock explanation failed ({BEDROCK_MODEL_ID}): {e}")
        return None


def build_distribution(probs, predicted_idx, lang, top_n=3):
    """Top-N classes as rows for the confidence bars.

    Percentages are rounded independently, so they will not always sum to 100.
    """
    count = min(top_n, int(probs.shape[0]))
    top = torch.topk(probs, count)
    labels = tr.GENRE_LABELS.get(lang, tr.GENRE_LABELS["en"])

    rows = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        key = label_encoder.inverse_transform([idx])[0]
        rows.append({
            "key": key,
            "label": labels.get(key, key),
            "pct": round(score * 100),
            "is_top": idx == predicted_idx,
        })
    return rows


# Prediction function 
def predict(image_bytes, lang="en"):
    try:
        image = load_image(image_bytes)
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            # Keep the whole distribution, not just the winner: the UI shows
            # the runner-up genres alongside the prediction.
            probs = torch.softmax(output, dim=1)[0]
            predicted_idx = int(torch.argmax(probs).item())

        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        distribution = build_distribution(probs, predicted_idx, lang)
        heatmap = generate_heatmap(input_tensor, predicted_idx)
        cam_image = overlay_heatmap(heatmap, image)

        cam_pil = Image.fromarray(cam_image)
        buf = io.BytesIO()
        cam_pil.save(buf, format='JPEG')
        gradcam_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Merge both frames into one image for the heatmap explanation
        genre_for_prompt = tr.GENRE_LABELS.get(lang, tr.GENRE_LABELS["en"]).get(
            predicted_label, predicted_label)
        merged = merge_images_side_by_side(image, cam_pil)
        g_explanation = get_vision_explanation(compress_image(merged), genre_for_prompt, lang, grad=True)
        o_explanation = get_vision_explanation(compress_image(image), genre_for_prompt, lang)
        return predicted_label, gradcam_base64, g_explanation, o_explanation, distribution
    except Exception as e:
        print("Prediction error:", e)
        return None, None, None, None, []

def view_context(lang, **extra):
    """Everything the template needs regardless of which state it renders."""
    labels = tr.GENRE_LABELS[lang]
    context = {
        "lang": lang,
        "direction": tr.DIRECTION[lang],
        "other_lang": tr.other(lang),
        "t": tr.UI[lang],
        "accent": tr.FALLBACK_ACCENT,
        "genre_accents": tr.GENRE_ACCENTS,
        "genre_chips": [(key, labels[key]) for key in tr.GENRE_ACCENTS],
        "prediction": None,
        "error": None,
    }
    context.update(extra)
    return context


def sample_tracks(playlist_id, limit=10):
    """Fetch a playlist and take a random handful of it."""
    tracks = get_playlist_tracks(playlist_id, YOUTUBE_API_KEY) if playlist_id else []
    return random.sample(tracks, min(limit, len(tracks))) if tracks else []


@app.route("/", methods=["GET", "POST"])
def index():
    # Language comes from the form on submit, the query string otherwise.
    lang = tr.normalize(request.form.get("lang") or request.args.get("lang"))

    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            return render_template("index.html", **view_context(lang))

        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("utf-8")

        try:
            # Fire-and-forget: a storage failure must not cost the user their result.
            invoke_lambda_to_store_image(
                image_bytes, bucket_name, lambda_function_name, region="us-east-1")

            prediction, gradcam_data, grad_text, original_text, distribution = predict(
                image_bytes, lang)

            if prediction is None:
                return render_template("index.html", **view_context(
                    lang, error=tr.UI[lang]["error"], image_data=image_data))

            playlists = PLAYLIST_MAP.get(prediction, {})
            original_tracks = sample_tracks(playlists.get("original"))
            khaleeji_tracks = sample_tracks(playlists.get("khaleeji"))

            return render_template("index.html", **view_context(
                lang,
                prediction=prediction,
                genre_label=tr.GENRE_LABELS[lang].get(prediction, prediction),
                prediction_info=tr.GENRE_DESCRIPTIONS[lang].get(prediction),
                accent=tr.GENRE_ACCENTS.get(prediction, tr.FALLBACK_ACCENT),
                distribution=distribution,
                grad_text=grad_text,
                original_text=original_text,
                image_data=image_data,
                gradcam_data=gradcam_data,
                original_tracks=original_tracks,
                khaleeji_tracks=khaleeji_tracks,
                track_total=len(original_tracks) + len(khaleeji_tracks),
                error=None if (original_tracks or khaleeji_tracks) else tr.UI[lang]["no_playlists"],
            ))

        except Exception as e:
            print("Request error:", e)
            return render_template("index.html", **view_context(
                lang, error=str(e), image_data=image_data))

    return render_template("index.html", **view_context(lang))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
