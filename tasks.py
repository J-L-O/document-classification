import msgpack
import os
from io import BytesIO

from PIL import Image
from celery import Celery

from classification import ClassificationTask

broker_address = os.environ.get('BROKER_ADDRESS', 'localhost')
app = Celery('wpi_demo', backend='rpc://', broker=f"pyamqp://guest@{broker_address}//")
app.conf.update(
    accept_content=['msgpack'],
    task_serializer='msgpack',
    result_serializer='msgpack',
)


@app.task(name='document_classification', base=ClassificationTask)
def classify(task_data):
    bytes = msgpack.unpackb(task_data)
    classify.initialize()

    image_data = BytesIO(bytes)
    image_data.seek(0)

    # print("Before trace")
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('192.168.2.104', port=6899, stdoutToServer=True, stderrToServer=True)
    # print("After trace")

    with Image.open(image_data) as decoded_image:
        decoded_image = decoded_image.convert('RGB')
        classification_result = classify.predict(decoded_image)
        # classification_result = {
        #     "predicted_class": "test",
        #     "confidence": 1.0
        # }

    return classification_result
