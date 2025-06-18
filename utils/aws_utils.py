import boto3
import base64
import json
import uuid
import io
from PIL import Image, UnidentifiedImageError
import imageio.v3 as iio

def load_image(image_bytes):
    """Try Pillow first, fallback to imageio for HEIC support (Mobile Pictures)."""
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        try:
            img_array = iio.imread(image_bytes)
            return Image.fromarray(img_array).convert("RGB")
        except Exception as e:
            raise ValueError(f"Unsupported image format: {e}")

def invoke_lambda_to_store_image(image_bytes, bucket_name, lambda_function_name, region="us-east-1"):
    """Invoke AWS Lambda to store the uploaded image in S3."""
    client = boto3.client("lambda", region_name=region)

    try:
        img = load_image(image_bytes)
        img_resized = img.resize((224, 224)) # preventing huge payload
        buffer = io.BytesIO()
        img_resized.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        resized_bytes = buffer.read()
        encoded_image = base64.b64encode(resized_bytes).decode("utf-8")
    except Exception as e:
        print(f"❌ Failed to process image: {e}")
        return

    filename = f"uploads/{uuid.uuid4()}.jpg" # Using uuid to create unique name
    payload = {
        "filename": filename,
        "image_data": encoded_image,
        "bucket": bucket_name
    }

    try:
        client.invoke(
            FunctionName=lambda_function_name,
            InvocationType="Event",
            Payload=json.dumps(payload)
        )
        print("✅ Lambda invoked to store image in S3.")
    except Exception as e:
        print(f"❌ Failed to invoke Lambda: {e}")