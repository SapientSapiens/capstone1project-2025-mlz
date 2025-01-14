import numpy as np

#import from tflite_runtime tflite
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

# loading the tflite model
interpreter = tflite.Interpreter(model_path='final_deployable_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# create preprocessor
preprocessor = create_preprocessor('xception', target_size=(299, 299))

# my image class list as derieved from my training notebook
bird_classes= ['Asian Green Bee-Eater',
 'Brown-Headed Barbet',
 'Cattle Egret',
 'Common Kingfisher',
 'Common Myna',
 'Common Rosefinch',
 'Common Tailorbird',
 'Coppersmith Barbet',
 'Forest Wagtail',
 'Gray Wagtail',
 'Hoopoe',
 'House Crow',
 'Indian Grey Hornbill',
 'Indian Peacock',
 'Indian Pitta',
 'Indian Roller',
 'Jungle Babbler',
 'Northern Lapwing',
 'Red-Wattled Lapwing',
 'Ruddy Shelduck',
 'Rufous Treepie',
 'Sarus Crane',
 'White Wagtail',
 'White-Breasted Kingfisher',
 'White-Breasted Waterhen']

#print("inside lambda_function........")

def predict(url):
    # feed image to preprocessor
    X = preprocessor.from_url(url)

    #aapplying this model to this image
    interpreter.set_tensor(input_index,  X.astype(np.float32))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    # checking the predictions
    prediction_probabilites = dict(zip(bird_classes, preds[0]))

    # Get the class with the highest probability
    max_class = max(prediction_probabilites, key=prediction_probabilites.get)

    # Get the corresponding probability
    max_probability = prediction_probabilites[max_class]
    # formatting the desired output
    pred_str = f"The bird appears to be a '{max_class}' with a probability of {max_probability}"

    return pred_str


def lambda_handler(event, context):
    #print("inside lambda_handler.........")
    url = event['url']
    result = predict(url)
    return result