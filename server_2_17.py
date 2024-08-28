from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from keras.layers import TFSMLayer
from PIL import Image
import io

app = Flask(__name__)

# Load the model
saved_model_path = 'saved_model/my_model'
loaded_layer = TFSMLayer(saved_model_path, call_endpoint='serving_default')

def prepare_image(image_bytes):
    # Convert the image to numpy array suitable for model input, assuming 28x28 pixels size
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalize to [0, 1]
    return image_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        img_file = request.files['file']
        image_bytes = img_file.read()
        image_array = prepare_image(image_bytes)

        # Run prediction
        predictions_dict = loaded_layer(image_array)
        predictions = predictions_dict['output_0']

        # Determine the predicted class and confidence
        predicted_class = int(np.argmax(predictions, axis=-1)[0])
        probabilities = tf.nn.softmax(predictions)  # Apply softmax to get probabilities
        confidence = float(probabilities[0][predicted_class])  # Confidence for the predicted class

        # Return the response as JSON
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
