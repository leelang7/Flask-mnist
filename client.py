import requests
import os

def upload_image(image_path):
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return

    url = "http://127.0.0.1:8888/predict"  # The server's URL

    try:
        # Open the image file in binary mode
        with open(image_path, 'rb') as img_file:
            # Prepare the file to send as part of the request
            files = {'file': (os.path.basename(image_path), img_file, 'multipart/form-data')}
            # Send the POST request to the server
            response = requests.post(url, files=files)

        # Check if the server response is successful
        if response.status_code == 200:
            # Print the server's plain text response
            print(response.text)
        else:
            # Print an error message with the status code
            print(f"Server returned status code {response.status_code}.")
            print("Response content:", response.text)

    except requests.exceptions.RequestException as e:
        # Handle any network-related errors
        print("An error occurred while making the request:", e)

# Specify the path to your image
image_path = "img/two.png"  # Replace with the actual path of your image
upload_image(image_path)
