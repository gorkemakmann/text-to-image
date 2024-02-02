import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

print('model is loading')
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
print('model loaded')
images = model.text_to_image("photograph of an astronaut swimming in the sea", batch_size=4)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)