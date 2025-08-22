FROM public.ecr.aws/lambda/python:3.10

# Pin keras-image-helper to version 0.0.1 for Python 3.10 compatibility
RUN pip install keras-image-helper==0.0.1

# Pin numpy and scipy to compatible versions
RUN pip install numpy==1.23.1
RUN pip install scipy==1.10.0 --no-deps

# Install TensorFlow Lite runtime (Python 3.10 wheel)
RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

# Pillow will be installed as a dependency of keras-image-helper==0.0.1

# Copy your model and handler script
COPY ./models/final_deployable_model.tflite .
COPY ./model-serving/lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]