import os
import json
import asyncio
import google.generativeai as genai
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

# Configure the API key
genai.configure(api_key="AIzaSyAZCLiF0kL0eV3N_Rc5pM4xbC1YbZxlHPI")
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

def detect_objects_with_bboxes(file_path, prompt):
    try:
        # Open the image using PIL (Python Imaging Library)
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            image = Image.open(BytesIO(image_data))  # Open image using PIL
            print("Image opened successfully.")

        # Convert the image to numpy array for processing
        img_array = np.array(image)
        
        # Configure the model
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro", generation_config=generation_config)
        print("Model configured successfully.")
        
        # Generate content (image processing and prompt together)
        response = model.generate_content([prompt, image])  # Pass the PIL Image object
        print("Content generated successfully.")
        
        # Check the raw response
        print("Raw Response:", response.text)  # Print the raw response here
        
        # Parse the response (handle invalid/empty responses)
        try:
            response_json = json.loads(response.text)  # Assuming response.text contains JSON
            print("Parsed Response:", response_json)
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
            return {'success': False, 'error': "Invalid JSON response."}
        
        # Extract the bounding boxes and labels from the response
        objects = response_json.get('objects', [])
        if objects:
            # Create a draw object to draw on the image
            draw = ImageDraw.Draw(image)
            
            for obj in objects:
                # Extract bounding box coordinates and label
                x1, y1, x2, y2 = obj.get('bbox', [0, 0, 0, 0])  # Bounding box coordinates
                label = obj.get('label', 'Unknown')  # Object label
                
                # Draw the bounding box and label
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, y1 - 10), label, fill="green")
            
            # Save the image with bounding boxes
            output_image_path = 'output_image_with_bboxes.jpg'
            image.save(output_image_path)
            print(f"Output image saved to {output_image_path}")
            
            return {
                'success': True,
                'description': response_json.get('text', 'No description available'),
                'raw_response': response_json,
                'output_image': output_image_path
            }
        else:
            return {'success': False, 'error': "No objects detected."}
    except Exception as e:
        return {'success': False, 'error': f"An error occurred: {e}"}

async def main():
    # File path for the input image
    file_path = "C:/Users/yasht/OneDrive/Desktop/Maitri_AI/Bundle_Object_Detection/kitchen1.jpeg"  # Ensure this path points to an actual image file
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Prompt for the generative model
    prompt = "Detect and describe the objects in this image. Also, give me the count and draw bounding boxes for each detected object."
    # Call the detection function
    result = detect_objects_with_bboxes(file_path, prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
