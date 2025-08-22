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

# The image URL to be sent as input
#data = {'url': 'https://media.gettyimages.com/id/135775020/photo/a-crow-quenches-its-thirst-with-water-leaking-from-a-pipe-at-the-zoo-in-lahore-24-june-2005.jpg?s=612x612&w=gi&k=20&c=gnQtw8CRKq0CYwCBiV5F8hPKHu3aFz978Vf9uG9DU4w='}
#data = {'url': 'https://static.theprint.in/wp-content/uploads/2023/03/Untitled-design-11-1.jpg?compress=true&quality=80&w=376&dpr=2.6'}
#data = {'url': 'https://birdcount.in/wp-content/uploads/2023/06/Lesser-Coucal-by-Muhammed-Rafi.jpeg'}
#data = {'url': 'https://media.gettyimages.com/id/2185461073/photo/grey-wagtail.jpg?s=2048x2048&w=gi&k=20&c=qX_-P2PM11CV6kZNe5EjKvKxQUP2_iq3p6-HhWx-hpU='}
# Sending a POST request to the Lambda function
result = requests.post(url, json=data).json()

# Printing the prediction result
print(result)
