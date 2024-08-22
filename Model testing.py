import requests

# Replace these with your Roboflow API details
api_key = 'rf_Tsh7GsD82fXA44W6oa6ZwV4p6b42'
endpoint_url = 'YOUR_ENDPOINT_URL'

# Path to the image you want to test
image_path = '/content/test.jpeg'

# Open the image file in binary mode
with open(image_path, 'rb') as image_file:
    # Prepare the payload for the POST request
    files = {'file': image_file}
    headers = {'Authorization': f'Bearer {api_key}'}

    # Send the POST request to the Roboflow API
    response = requests.post(endpoint_url, files=files, headers=headers)

    # Print the response
    result = response.json()
    print(result)
