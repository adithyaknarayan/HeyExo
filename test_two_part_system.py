#!/usr/bin/env python3
"""
Test script to demonstrate the two-part prompt system with separate files.
This script shows how the system:
1. Loads prompts from separate const files
2. First identifies the task from an image
3. Dynamically injects that information into the assistance adjustment prompt
"""

import sys
from pathlib import Path
from dialogue_system import ExoskeletonDialogueSystem

def test_two_part_system():
    """Test the two-part prompt system with a sample image."""
    
    # Check if we have a sample image
    sample_images = [
        "data/stairs.webp",
        "data/learnimg_stairs.max-752x423.jpg"
    ]
    
    image_path = None
    for img in sample_images:
        if Path(img).exists():
            image_path = img
            break
    
    if not image_path:
        print("No sample images found. Please add an image to the data/ directory.")
        return
    
    print("="*60)
    print("TESTING TWO-PART PROMPT SYSTEM WITH SEPARATE FILES")
    print("="*60)
    print(f"Using image: {image_path}")
    print("Loading prompts from: const_task.py and const_assistance.py")
    print()
    
    try:
        # Initialize the dialogue system
        dialogue_system = ExoskeletonDialogueSystem()
        
        # Set the image
        dialogue_system.set_image(image_path)
        
        # Test task identification only
        print("STEP 1: Task Identification")
        print("-" * 30)
        task_result = dialogue_system.identify_task()
        print(f"Identified Task: {task_result['task']}")
        print(f"Terrain Difficulty: {task_result['terrain_difficulty']}")
        print(f"Confidence: {task_result['confidence']:.2f}")
        print()
        
        # Test full analysis with sample feedback
        sample_feedback = "Need more help"
        print("STEP 2: Full Analysis with User Feedback")
        print("-" * 40)
        print(f"User Feedback: '{sample_feedback}'")
        print()
        
        result = dialogue_system.analyze_feedback(sample_feedback)
        
        print("FINAL RESULT:")
        print("-" * 20)
        print(f"Task: {result['task']} (confidence: {result['confidence']:.2f})")
        print(f"Terrain Difficulty: {result['terrain_difficulty']}")
        print(f"Initial Assistance: {result['initial_assistance']}%")
        print(f"Previous Assistance: {dialogue_system.previous_assistance}%")
        print(f"Delta: {result['assistance_delta']:+.1f}%")
        print(f"Final Assistance: {result['final_assistance_level']}% (calculated: {dialogue_system.previous_assistance} + {result['assistance_delta']})")
        print(f"Reasoning: {result['reasoning']}")
        
        print("\n" + "="*60)
        print("TWO-PART SYSTEM WITH SEPARATE FILES TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(test_two_part_system())
