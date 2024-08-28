import requests

def predict_image(image_path):
    url = "http://localhost:5001/predict"
    
    with open(image_path, 'rb') as img_file:
        files = {'file': img_file}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        print(f"Predicted class: {data['predicted_class']}, Confidence: {data['confidence']}")
    else:
        print("Error:", response.json())

# Example usage:
predict_image('img/eight.png')  # Provide path to an MNIST-like image
