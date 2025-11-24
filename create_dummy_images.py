import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_dummy_images():
    """Create dummy images for training data."""
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Flat ground image
    flat_img = Image.new('RGB', (512, 512), color='lightgreen')
    draw = ImageDraw.Draw(flat_img)
    draw.rectangle([50, 200, 462, 300], fill='gray', outline='black', width=2)
    draw.text((200, 400), "Flat Ground", fill='black')
    flat_img.save("data/flat_ground.jpg")
    
    # Stairs up image
    stairs_img = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(stairs_img)
    # Draw stairs
    for i in range(8):
        y = 300 - i * 20
        x = 100 + i * 30
        draw.rectangle([x, y, x + 30, y + 20], fill='gray', outline='black')
    draw.text((200, 400), "Stairs Up", fill='black')
    stairs_img.save("data/stairs_up.jpg")
    
    # Stairs down image
    stairs_down_img = Image.new('RGB', (512, 512), color='lightcoral')
    draw = ImageDraw.Draw(stairs_down_img)
    # Draw stairs going down
    for i in range(8):
        y = 200 + i * 20
        x = 100 + i * 30
        draw.rectangle([x, y, x + 30, y + 20], fill='gray', outline='black')
    draw.text((200, 400), "Stairs Down", fill='black')
    stairs_down_img.save("data/stairs_down.jpg")
    
    # Uphill image
    uphill_img = Image.new('RGB', (512, 512), color='lightyellow')
    draw = ImageDraw.Draw(uphill_img)
    # Draw slope
    points = [(50, 400), (200, 300), (350, 200), (462, 150)]
    draw.polygon(points, fill='brown', outline='black')
    draw.text((200, 450), "Uphill", fill='black')
    uphill_img.save("data/uphill.jpg")
    
    # Steep uphill image
    steep_uphill_img = Image.new('RGB', (512, 512), color='lightpink')
    draw = ImageDraw.Draw(steep_uphill_img)
    # Draw steeper slope
    points = [(50, 400), (150, 250), (250, 150), (350, 100), (462, 80)]
    draw.polygon(points, fill='darkbrown', outline='black')
    draw.text((200, 450), "Steep Uphill", fill='black')
    steep_uphill_img.save("data/uphill_steep.jpg")
    
    print("Created dummy images:")
    print("- data/flat_ground.jpg")
    print("- data/stairs_up.jpg") 
    print("- data/stairs_down.jpg")
    print("- data/uphill.jpg")
    print("- data/uphill_steep.jpg")

if __name__ == "__main__":
    create_dummy_images()

