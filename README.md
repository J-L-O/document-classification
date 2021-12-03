# document-demo

Defines a celery task that applies a model trained with the code from [this](https://github.com/J-L-O/Unsupervised-Classification) repository to an input image.

Expects two environment variables:

- CLASSIFICATION_DEVICE: The device to use for inference. Positive numbers specify a GPU, any negative number means CPU inference
- CLASSIFICATION_CONFIG_PATH: Path to a config folder containing
    1. model.pth.tar: The trained classification model
    2. config.yml: The SCAN config used to train the model
    3. env.yml: The env.yml from the training repository
    4. classes.txt: A text file containing the class names in the same order as the model outputs them. The entries must be separated by newlines.
