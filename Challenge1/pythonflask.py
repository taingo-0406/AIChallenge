from flask import Flask, render_template, request, jsonify
import base64
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# print(tf.__version__)

model = tf.keras.models.load_model('keras3.h5')
model.make_predict_function()


@app.route('/')
def index():

    return render_template('index2.html')

@app.route('/recognize', methods = ['POST'])
def recognize():

    if request.method == 'POST':
        print("Receive image and predict what it is")
        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        with open("temp.jpg", "wb") as temp:
            temp.write(imgBytes)

        # with open('class_names.txt') as f:
        #     classes = f.readlines()
        # classes = [c.replace('\n', '').replace(' ', '_') for c in classes]

        # image = cv2.imread('temp.jpg')
        # image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image_prediction = np.reshape(image_gray, (28,28,1))
        # image_prediction = (255 - image_prediction.astype('float')) / 255

        # # prediction = np.argmax(model.predict(np.array([image_prediction])), axis = -1)

        # prediction = model.predict(np.expand_dims(image_prediction, axis=0))[0]
        # ind = (-prediction).argsort()[:5]
        # latex = [classes[x] for x in ind]

        #Chay chay prediction

        CATEGORIES = ["Square", "Circle"]

        def prepare(filepath):
            IMG_SIZE = 28
            img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        model = tf.keras.models.load_model("keras3.h5")
        model.make_predict_function()

        prediction = model.predict([prepare('temp.jpg')])

    # return jsonify({
    #     'prediction' : str(latex),
    #     'status' : True
    # })

    return jsonify({
            "prediction": "Ok, I see it is a " +CATEGORIES[int(prediction[0][0])],
            "status": True
        })

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(debug = True)