import os
import json
import asyncio
import google.generativeai as genai
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import re

# Configure the API key
genai.configure(api_key="AIzaSyAZCLiF0kL0eV3N_Rc5pM4xbC1YbZxlHPI")

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

def extract_relative_position(text):
    """
    Extract relative position information from text description.
    Returns tuple of (horizontal_pos, vertical_pos, size_hint)
    """
    text = text.lower()
    
    # Initialize position hints
    horizontal_pos = 0.5  # Default to center
    vertical_pos = 0.5   # Default to middle
    size_hint = 0.3      # Default size (30% of image)
    
    # Horizontal position processing
    if any(term in text for term in ['left side', 'left of', 'left part']):
        horizontal_pos = 0.2
    elif any(term in text for term in ['right side', 'right of', 'right part']):
        horizontal_pos = 0.8
    elif 'center' in text:
        horizontal_pos = 0.5
    
    # Vertical position processing
    if any(term in text for term in ['top', 'upper']):
        vertical_pos = 0.2
    elif any(term in text for term in ['bottom', 'lower']):
        vertical_pos = 0.8
    elif 'middle' in text:
        vertical_pos = 0.5
        
    # Size hints processing
    if any(term in text for term in ['large', 'big']):
        size_hint = 0.4
    elif any(term in text for term in ['small', 'tiny']):
        size_hint = 0.2
    
    return (horizontal_pos, vertical_pos, size_hint)

def relative_to_absolute_coords(image_size, rel_x, rel_y, size_hint):
    """Convert relative positions to absolute pixel coordinates for bounding box"""
    width, height = image_size
    box_width = int(width * size_hint)
    box_height = int(height * size_hint)
    
    center_x = int(width * rel_x)
    center_y = int(height * rel_y)
    
    # Calculate box coordinates
    x1 = max(0, center_x - box_width//2)
    y1 = max(0, center_y - box_height//2)
    x2 = min(width, center_x + box_width//2)
    y2 = min(height, center_y + box_height//2)
    
    return (x1, y1, x2, y2)

def parse_objects_from_response(response_text, image_size):
    """Parse the response text to extract objects and their relative positions"""
    objects = []
    
    # Split into sentences and filter out empty ones
    sentences = [s.strip() for s in re.split('[.!?]', response_text) if s.strip()]
    
    for sentence in sentences:
        # Skip sentences that don't match our expected format
        if 'is located in the' not in sentence.lower():
            continue
            
        try:
            # Split the sentence into object name and position description
            parts = sentence.lower().split('is located in the')
            if len(parts) != 2:
                continue
                
            obj_name = parts[0].replace('the ', '').strip()
            position_desc = parts[1].strip()
            
            # Get relative positions
            rel_x, rel_y, size = extract_relative_position(position_desc)
            
            # Convert to absolute coordinates
            bbox = relative_to_absolute_coords(image_size, rel_x, rel_y, size)
            
            objects.append({
                'name': obj_name.strip(),
                'bbox': bbox,
                'description': position_desc
            })
            
        except Exception as e:
            print(f"Error parsing sentence: {sentence}")
            print(f"Error details: {str(e)}")
            continue
    
    return objects

def draw_bounding_boxes(image, objects):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    
    for obj in objects:
        bbox = obj['bbox']
        
        # Draw rectangle
        draw.rectangle(bbox, outline='red', width=2)
        
        # Draw label background
        label = obj['name']
        label_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label)
        draw.rectangle(label_bbox, fill='red')
        
        # Draw label text
        draw.text((bbox[0], bbox[1] - 20), label, fill='white')
    
    return image

def detect_objects(file_path):
    try:
        # Open the image using PIL
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            image = Image.open(BytesIO(image_data))
            print("Image opened successfully.")

        # Configure the model
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro", 
                                    generation_config=generation_config)
        print("Model configured successfully.")

        # Enhanced structured prompt
        prompt = """
        Analyze this image and provide detailed object detection in the following EXACT format:

        Rules for describing each object:
        1. Use this EXACT pattern:
           'The [specific object name] is located in the [position] of the image, [size] size'

        2. Position MUST use these EXACT terms only:
           - Horizontal: 'left side', 'right side', 'center'
           - Vertical: 'top', 'bottom', 'middle'
           Example: 'left side top', 'center middle', 'right side bottom'

        3. Size MUST be either 'small' or 'large' only

        4. Object names MUST be specific and detailed:
           - Include color, type, and distinguishing features
           - Example: 'dark blue Honda sedan' instead of just 'car'
           - Example: 'metal street lamp post' instead of just 'lamp'

        Example correct responses:
        'The silver Toyota Camry is located in the left side bottom of the image, large size'
        'The black metal street lamp is located in the right side top of the image, small size'
        'The red brick building is located in the center middle of the image, large size'

        Please detect and describe ALL visible objects using ONLY this format.
        After listing all objects, provide a total count of objects detected.
        """

        # Generate content
        response = model.generate_content([prompt, image])
        print("Content generated successfully.")

        # Parse objects and their positions
        detected_objects = parse_objects_from_response(response.text, image.size)
        
        # Create a copy of the image for drawing
        annotated_image = image.copy()
        
        # Draw bounding boxes if objects were detected
        if detected_objects:
            annotated_image = draw_bounding_boxes(annotated_image, detected_objects)
            
            # Save the annotated image
            output_dir = "static/outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "annotated_" + os.path.basename(file_path))
            annotated_image.save(output_path)
            print(f"Annotated image saved to: {output_path}")

        return {
            'success': True,
            'description': response.text,
            'objects': detected_objects,
            'annotated_image_path': output_path if detected_objects else None
        }

    except Exception as e:
        return {'success': False, 'error': f"An error occurred: {e}"}

async def main():
    # File path for the input image
    file_path = "static/uploads/images/City.jpg"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Call the detection function
    result = detect_objects(file_path)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())