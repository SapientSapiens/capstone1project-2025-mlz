import numpy as np
import base64
import io
import logging
from PIL import Image, UnidentifiedImageError
import imghdr
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
from scipy.special import softmax

# --- Logging setup for AWS Lambda (console and logs go to CloudWatch) ---
def setup_logger():
    logger = logging.getLogger("bird_classifier")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

# Load TFLite model globally

# uncomment for running in the local machine (this is in accordance with the repo directory structure.)
# interpreter = tflite.Interpreter(model_path='../models/final_deployable_model.tflite') 

# the model and this script will be at the root directory of the container. Comment this line if you uncomment the above one.
interpreter = tflite.Interpreter(model_path='final_deployable_model.tflite') 

interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Create preprocessor globally
preprocessor = create_preprocessor('xception', target_size=(299, 299))

# Bird classes
bird_classes = [
    'Asian Green Bee-Eater', 'Brown-Headed Barbet', 'Cattle Egret', 'Common Kingfisher',
    'Common Myna', 'Common Rosefinch', 'Common Tailorbird', 'Coppersmith Barbet',
    'Forest Wagtail', 'Gray Wagtail', 'Hoopoe', 'House Crow', 'Indian Grey Hornbill',
    'Indian Peacock', 'Indian Pitta', 'Indian Roller', 'Jungle Babbler',
    'Northern Lapwing', 'Red-Wattled Lapwing', 'Ruddy Shelduck', 'Rufous Treepie',
    'Sarus Crane', 'White Wagtail', 'White-Breasted Kingfisher', 'White-Breasted Waterhen'
]

def handle_unsupported_format_image(image_data: bytes):
    """Robust image handling with lightweight error reporting"""
    try:
        # First try to open the image directly
        return Image.open(io.BytesIO(image_data))
        
    except UnidentifiedImageError:
        # Pillow failed to open - try to determine why
        detected_format = imghdr.what(None, image_data)
        
        if detected_format:
            try:
                # Second attempt: force open with format hint
                return Image.open(io.BytesIO(image_data), formats=[detected_format])
            except Exception:
                # Format recognized but still can't open
                msg  = f"Unsupported image format: {detected_format.upper()}. Please convert to JPEG or PNG."
                logger.error(msg)
                return msg
        else:
            if len(image_data) < 12:
                msg = "Image data too small (may be empty or corrupted)"
                logger.error(msg)
                return msg
                
            # Provide minimal header info
            header = image_data[:16].hex()
            msg = f"Unrecognized image format. First 16 bytes: {header}... Please use JPEG or PNG."
            logger.error(msg)
            return msg


def predict(image_base64: str) -> dict:
    try:
        # Decode Base64 image
        image_data = base64.b64decode(image_base64)
        logger.info("Image data decoded from base64.")

        # Handle image opening
        image_result = handle_unsupported_format_image(image_data)
        
        # Check if we got an error string
        if isinstance(image_result, str):
            logger.error(f"Image processing error: {image_result}")
            return {"error": image_result}
        
        # Convert to model-ready tensor
        X = preprocessor.convert_to_tensor(image_result)
        logger.info("Image converted to model tensor.")

        # Run TFLite model
        interpreter.set_tensor(input_index, X.astype(np.float32))
        interpreter.invoke()
        logits = interpreter.get_tensor(output_index)[0]  # Get 1D array

        # Convert logits to probabilities
        probabilities = softmax(logits)
        prediction_probs = dict(zip(bird_classes, probabilities))

        # Get the class with the highest probability
        max_class = max(prediction_probs, key=prediction_probs.get)
        max_probability = prediction_probs[max_class]

        logger.info(f"Prediction: {max_class} ({max_probability*100:.2f}%)")
        # return f"The bird appears to be a '{max_class}' with a probability of {max_probability * 100:.2f}%"
        # Format output as JSON for client-side logic
        return {
            "predicted_class": max_class,
            "probability": float(max_probability)  # ensure JSON serializable
        }

    except Exception as e:
        logger.error(f"Exception in prediction: {str(e)}")
        return {"error": str(e)}



def lambda_handler(event, context):
    image_base64 = event.get('image_base64')
   
    if not image_base64:
        msg = "No image_base64 found in the request"
        logger.error(msg)
        return {"error": msg}
    
    result = predict(image_base64)
    return result