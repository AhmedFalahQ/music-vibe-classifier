import boto3
import base64
import json
import logging
import uuid
import io

from utils.images import load_image

logger = logging.getLogger(__name__)

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
        logger.warning("Could not process the upload for archiving", exc_info=True)
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
        logger.info("Lambda invoked to archive the upload as %s", filename)
    except Exception as e:
        logger.warning("Failed to invoke the archiving Lambda", exc_info=True)
