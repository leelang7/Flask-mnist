from flask import Flask, jsonify, request, render_template, redirect, url_for
import tensorflow as tf
import PIL.Image as Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def work(img, model):
    pred = model.predict(img)
    pred = pred[0]
    idx = tf.math.argmax(pred)
    confidence = tf.math.reduce_max(pred)
    return idx, confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result_string = "please upload an image or input image URL"

    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = Image.open(filepath)
            img = tf.keras.utils.img_to_array(img)
            img = tf.expand_dims(img, axis=0)

            idx, confidence = work(img, model)
            result_string = f"This is number {idx} with {confidence:.2f} confidence"

    return render_template("index.html", result_string=result_string)

if __name__ == "__main__":
    model = tf.keras.models.load_model("mymodel")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host="127.0.0.1", port=8888, debug=True)
