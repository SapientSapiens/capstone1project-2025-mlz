import requests
import base64
from PIL import Image, UnidentifiedImageError
import io

print("Testing the model deployed at AWS Lambda Service...")

# The URL where the Lambda function is exposed
url = 'https://vnj365geif.execute-api.eu-north-1.amazonaws.com/beta/predict'
#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# ---- Load & resize image before sending ----
image_path = "./test_images/wagtail-grey033.jpg"

try:
    img = Image.open(image_path)
    img = img.resize((299, 299)).convert("RGB")   # match model input size and also handles input non JPG formats

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")   # compress as JPEG
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    data = {'image_base64': img_b64}

    # Sending a POST request to the Lambda function
    result = requests.post(url, json=data).json()
    print(result)

except UnidentifiedImageError:
    print(f"❌ Could not open {image_path}. Unsupported format. Try converting to JPG/PNG.")
except Exception as e:
    print(f"⚠️ Unexpected error: {e}")
