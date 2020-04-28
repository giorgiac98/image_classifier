import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json

def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    image = np.asarray(img)
    proc_img = process_image(image)

    pred = model.predict(np.expand_dims(proc_img, axis=0))

    top_k_index = np.argsort(pred[0])[-top_k:]
    top_k_pred = [pred[0][i] for i in top_k_index]
    top_k_label = [str(i+1) for i in top_k_index]

    return top_k_pred, top_k_label


parser = argparse.ArgumentParser(description='Predict flower label')
parser.add_argument('image_path', metavar='image_path', type=str,
                    help='path to image you want to predict')
parser.add_argument('saved_model', metavar='saved_model', type=str,
                    help='saved model that will be used for the prediction')
parser.add_argument('--top_k', dest='k', type=int, action='store', default=1,
                    help='Return the top K most likely classes')
parser.add_argument('--category_names', dest='category_names', type=str, action='store', default="label_map.json",
                    help='Path to a JSON file mapping labels to flower names')

args = parser.parse_args()

with open(args.category_names, 'r') as f:
    class_names = json.load(f)

print("Loading model...")
reloaded_keras_model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer})

print("Making predictions...")
top_k_pred, top_k_label = predict(args.image_path, reloaded_keras_model, args.k)
class_label = [class_names[i] for i in top_k_label]

for i in range(len(top_k_pred)):
    print("Class probability {}: {:.2f} %".format(class_label[i], top_k_pred[i]*100))
