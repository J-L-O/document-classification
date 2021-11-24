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

    with Image.open(image_data) as decoded_image:
        decoded_image = decoded_image.convert('RGB')
        classification_result = classify.predict(decoded_image)

    return classification_result
