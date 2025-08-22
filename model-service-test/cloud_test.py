import requests
import base64

print("Testing the model deployed at AWS Lambda Service...")

# The URL where the AWS Lambda function is exposed through the AWS API Gateway
#url = 'https://mb74pfois9.execute-api.eu-north-1.amazonaws.com/test'
url =  'https://vnj365geif.execute-api.eu-north-1.amazonaws.com/beta/predict'

# The image URL to be sent as input. You can use your own image url but it should be a bird from the mentioned 25 species. (obsolete functionality)

# image url of the House Crow
# data = {'url': 'https://media.gettyimages.com/id/135775020/photo/a-crow-quenches-its-thirst-with-water-leaking-from-a-pipe-at-the-zoo-in-lahore-24-june-2005.jpg?s=612x612&w=gi&k=20&c=gnQtw8CRKq0CYwCBiV5F8hPKHu3aFz978Vf9uG9DU4w='}

# image url of the Sarus Crane
# data = {'url': 'https://static.theprint.in/wp-content/uploads/2023/03/Untitled-design-11-1.jpg?compress=true&quality=80&w=376&dpr=2.6'}

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

