import requests
import base64

print("Testing the model deployed at AWS Lambda Service...")

# The URL where the AWS Lambda function is exposed through the AWS API Gateway
#url = 'https://mb74pfois9.execute-api.eu-north-1.amazonaws.com/test'
url =  'https://vnj365geif.execute-api.eu-north-1.amazonaws.com/beta/predict'

# base64 image of the House Crow
'''with open("../test_images/house_crow_image.jpg", "rb") as f_in:
    img_b64 = base64.b64encode(f_in.read()).decode("utf-8")'''

# base64 image of the White Wagtail
'''with open("../test_images/Crier+Header+Photos.webp", "rb") as f_in:
    img_b64 = base64.b64encode(f_in.read()).decode("utf-8")'''

# base64 image of the White Wagtail
with open("../test_images/image_sarus_crane.avif", "rb") as f_in:
    img_b64 = base64.b64encode(f_in.read()).decode("utf-8")

data = {'image_base64': img_b64}

# Sending a POST request to the Lambda function
result = requests.post(url, json=data).json()

# Printing the prediction result
print(result)