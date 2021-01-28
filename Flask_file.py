from flask import Flask, make_response, request,jsonify
from werkzeug.utils import secure_filename
import json
import pickle as pk
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
import cv2
import os
import numpy as np
import tensorflow as tf

from keras.applications.vgg16 import VGG16
vgg16 = VGG16(weights='imagenet')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


# from Keras GitHub
def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results
# vgg16 = load_model('vgg16.h5')
with open('vgg16_cat_list.pk', 'rb') as f:
    cat_list2 = pk.load(f)


def prepare_image_frame(img):
    img=cv2.resize(img,(224, 224)).astype(np.float32)
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def car_categories_gate_video(img):
#     urllib.request.urlretrieve(image_path, 'save.jpg') # or other way to upload image
    img = prepare_image_frame(img)
    out = vgg16.predict(img)
    top = get_predictions(out, top=5)
#     print(cat_list2)
    for j in top[0]:
        if j[0:2] in cat_list2:
#             print(j[0:2])
            return True
    return False


classes=['damaged','undamaged']

app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1> Upload file Which you want to predict and click on Submit</h1>

                <form action="/prediction" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
            </body>
        </html>
    """

@app.route('/prediction', methods=["POST"])
def transform_view():
    result = {'response code': 200, 'message': "", 'data': {"car_or_not":"", "damage_or_not":"", "manual":0}}



    try:

        if 'data_file' not in request.files:
            print('hi')
            result['message'] = "No file"
            return jsonify(result)
        file = request.files['data_file']
        if file.filename == '':
            result['message'] = "No image selected for uploading"
            return jsonify(result)
        if file and allowed_file(file.filename):
            result['message'] = "Success"
            for filename in os.listdir('static/uploads/'):
                os.remove('static/uploads/' + filename)
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            img = cv2.imread('static/uploads/' + filename)
            result1 = car_categories_gate_video(img)

            if result1:
                result['data']['car_or_not'] = "Validation complete Its a Car - proceed to damage evaluation"
                model = tf.keras.models.load_model('Car_damage_model2.h5')
                img = cv2.resize(img, (224, 224)).astype(np.float32)
                img = np.expand_dims(img, axis=0)
                pred = model.predict(img)
                res = pred[0]

                idx = np.argmax(res)
                label = classes[idx]
                prob=res[idx] * 100

                label = "{}: {:.3f}%".format(label, prob)
                result['data']['damage_or_not'] = label
                if prob<=70:
                    result['data']['manual'] = 1

            else:
                result['data']['car_or_not'] = "Are you sure this is a picture of your car? "

            return jsonify(result)
    except:
        result['message'] = "Incorrect File Information"
        return jsonify(result)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)