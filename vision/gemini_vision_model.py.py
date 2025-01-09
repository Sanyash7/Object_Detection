import os
import json
import asyncio
import google.generativeai as genai
from io import BytesIO
from PIL import Image

# Configure the API key
genai.configure(api_key="API-Key")

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

def detect_objects(file_path, prompt):
    try:
        # Open the image using PIL (Python Imaging Library)
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            image = Image.open(BytesIO(image_data))  # Open image using PIL
            print("Image opened successfully.")

        # Configure the model
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro", generation_config=generation_config)
        print("Model configured successfully.")

        # Generate content (image processing and prompt together)
        response = model.generate_content([prompt, image])  # Pass the PIL Image object
        print("Content generated successfully.")

        # Parse the response
        response_json = response.text  # Assuming response.text contains JSON
        print(response_json)
        objects = response_json.get('objects', [])

        if objects:
            # Returning results without drawing bounding boxes
            return {
                'success': True,
                'description': response_json.get('text', 'No description available'),
                'raw_response': response_json
            }
        else:
            return {'success': False, 'error': "No objects detected."}

    except Exception as e:
        return {'success': False, 'error': f"An error occurred: {e}"}

async def main():
    # File path for the input image
    file_path = "static/uploads/images/City.jpg"  # Ensure this path points to an actual image file

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Prompt for the generative model
    prompt = "Detect and describe the objects in this image.Also give me the count."

    # Call the detection function
    result = detect_objects(file_path, prompt)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
