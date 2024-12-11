import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def calculate_distance(image_path):
    # Load the image
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print("Image file not found. Please check the path and try again.")
        return

    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("Click on two points")
    plt.axis("on")
    
    # Get two points from user clicks
    print("Please click on two points in the image.")
    points = plt.ginput(2, timeout=60)
    plt.close(fig)

    if len(points) < 2:
        print("Insufficient points clicked. Please try again.")
        return

    # Extract coordinates
    (x1, y1), (x2, y2) = points
    pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    print(f"Distance in pixels: {pixel_distance:.2f}")

    # Ask for real-world distance
    real_world_distance = float(input("Enter the real-world distance between these points in millimeters: "))
    if real_world_distance <= 0:
        print("Real-world distance must be greater than zero.")
        return

    # Calculate pixels-to-millimeters conversion factor
    pixels_to_mm = real_world_distance / pixel_distance
    print(f"Pixels to millimeters conversion factor: {pixels_to_mm:.4f} mm/pixel")

    return pixels_to_mm

# Uncomment the line below to run the function directly
# pixels_to_mm = calculate_distance()
