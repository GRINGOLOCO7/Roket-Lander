import numpy as np
import cv2

# Create a landing background (ground and sky)
def create_landing_background(width=800, height=768):
    # Create a blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky gradient (blue to dark blue)
    for y in range(height):
        # Calculate blue intensity based on height (darker at top, lighter at bottom)
        blue = min(255, int(180 + (y / height) * 75))
        # Lighter at horizon
        green = min(255, int(100 + (y / height) * 100))
        # Almost no red except near horizon
        red = min(255, int(50 + (y / height) * 50))
        
        cv2.line(img, (0, y), (width, y), (blue, green, red), 1)
    
    # Ground/surface (gray)
    ground_y = height - 30  # Position ground near bottom
    cv2.rectangle(img, (0, ground_y), (width, height), (70, 70, 70), -1)
    
    # Add some surface details/texture
    for i in range(20):
        x = np.random.randint(0, width)
        w = np.random.randint(20, 100)
        cv2.rectangle(img, (x, ground_y), (x + w, ground_y + 5), (90, 90, 90), -1)
    
    # Add stars in the sky
    for _ in range(100):
        x = np.random.randint(0, width)
        y = np.random.randint(0, ground_y - 100)
        brightness = np.random.randint(150, 255)
        cv2.circle(img, (x, y), 1, (brightness, brightness, brightness), -1)
    
    # Save the image
    cv2.imwrite("landing.jpg", img)
    print("Created landing.jpg")

# Create a hover background (abstract environment)
def create_hover_background(width=800, height=768):
    # Create a blank image with gradient
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient background (dark to light)
    for y in range(height):
        # Calculate color based on height
        blue = min(255, int(50 + (y / height) * 150))
        green = min(255, int(30 + (y / height) * 100))
        red = min(255, int(10 + (y / height) * 70))
        
        cv2.line(img, (0, y), (width, y), (blue, green, red), 1)
    
    # Add grid lines for perspective
    grid_spacing = 50
    for i in range(0, width, grid_spacing):
        alpha = 0.3  # Transparency
        color = (100, 100, 100)
        cv2.line(img, (i, 0), (i, height), color, 1)
    
    for i in range(0, height, grid_spacing):
        alpha = 0.3  # Transparency
        color = (100, 100, 100)
        cv2.line(img, (0, i), (width, i), color, 1)
    
    # Add some floating objects/clouds
    for _ in range(10):
        x = np.random.randint(50, width-100)
        y = np.random.randint(50, height-100)
        size = np.random.randint(20, 60)
        color = (
            np.random.randint(180, 230),
            np.random.randint(180, 230),
            np.random.randint(180, 230)
        )
        cv2.circle(img, (x, y), size, color, -1)
        # Add some blur to make it look like clouds
        roi = img[y-size:y+size, x-size:x+size]
        if roi.size > 0:  # Ensure ROI is valid
            cv2.GaussianBlur(roi, (15, 15), 0, roi)
    
    # Save the image
    cv2.imwrite("hover.jpg", img)
    print("Created hover.jpg")

if __name__ == "__main__":
    create_landing_background()
    create_hover_background()
    print("Background images created successfully!") 