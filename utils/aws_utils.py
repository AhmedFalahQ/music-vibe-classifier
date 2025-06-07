import boto3
import base64
from botocore.exceptions import ClientError
import json
import uuid
def invoke_lambda_to_store_image(image_bytes, bucket_name, lambda_function_name, region="us-east-1"):
    """Invoke AWS Lambda to store the uploaded image in S3 for future training."""
    client = boto3.client("lambda", region_name=region)

    # Create a unique filename
    filename = f"uploads/{uuid.uuid4()}.jpg"

    # Encode image to base64 to safely pass in payload
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "filename": filename,
        "image_data": encoded_image,
        "bucket": bucket_name
    }

    try:
        response = client.invoke(
            FunctionName=lambda_function_name,
            InvocationType="Event",
            Payload=json.dumps(payload)
        )
        print("✅ Lambda invoked to store image in S3.")
    except Exception as e:
        print(f"❌ Failed to invoke Lambda: {e}")