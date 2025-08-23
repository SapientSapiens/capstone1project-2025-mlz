import requests
import base64

# The URL where the Lambda function is exposed from the container running in local machine
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# base64 image of the House Crow
'''with open("../test_images/crow-greyscale.jpg", "rb") as f_in:
    img_b64 = base64.b64encode(f_in.read()).decode("utf-8")'''

# base64 image of the Sarus Crane
'''with open("../test_images/image_sarus_crane.avif", "rb") as f_in:
    img_b64 = base64.b64encode(f_in.read()).decode("utf-8")'''
    
# base64 image of the Forest Wagtail
'''with open("../test_images/pngtree-vibrant-forest-wagtail-bird-illustration-on-white-backgroun-image_16211306.jpg", "rb") as f_in:
    img_b64 = base64.b64encode(f_in.read()).decode("utf-8")'''

# base64 image of the Coppersmith Barbet
'''with open("../test_images/coppersmith-barbet-prerna-jain.jpg", "rb") as f_in:
    img_b64 = base64.b64encode(f_in.read()).decode("utf-8")'''

# base64 image of the Test Bird
with open("../test_images/house_crow_image.jpg", "rb") as f_in:
    img_b64 = base64.b64encode(f_in.read()).decode("utf-8")


data = {'image_base64': img_b64}

result = requests.post(url, json=data).json()

# Printing the prediction result
print(result)
