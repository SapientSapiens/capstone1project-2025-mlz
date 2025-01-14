import requests

# The URL where the Lambda function is exposed
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# The image URL to be sent as input
data = {'url': 'https://media.gettyimages.com/id/135775020/photo/a-crow-quenches-its-thirst-with-water-leaking-from-a-pipe-at-the-zoo-in-lahore-24-june-2005.jpg?s=612x612&w=gi&k=20&c=gnQtw8CRKq0CYwCBiV5F8hPKHu3aFz978Vf9uG9DU4w='}

# Sending a POST request to the Lambda function
result = requests.post(url, json=data).json()

# Printing the prediction result
print(result)

