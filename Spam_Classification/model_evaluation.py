import requests
import json

# Define the URL of the API endpoint
url = "http://10.129.205.188:8000/api/upload"

# Path to the model file you want to upload
model_file_path = "spam_detection_model.joblib"

# Open the file in binary mode and send the POST request
with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

# Pretty print the response from the server
print(json.dumps(response.json(), indent=4))