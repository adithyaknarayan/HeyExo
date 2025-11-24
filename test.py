import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import sys

def load_prompt_from_const():
    """Load the prompt from const.py file."""
    const_path = Path(__file__).parent / "const.py"
    if not const_path.exists():
        raise FileNotFoundError("const.py file not found")
    
    with open(const_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the prompt content (everything between the first and last triple quotes)
    start_marker = '"""'
    end_marker = '"""'
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find start marker in const.py")
    
    start_idx += len(start_marker)
    end_idx = content.rfind(end_marker)
    if end_idx == -1 or end_idx <= start_idx:
        raise ValueError("Could not find end marker in const.py")
    
    prompt = content[start_idx:end_idx].strip()
    return prompt
def analyze_exoskeleton_scene(image_path: str, user_feedback: str, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """
    Analyze exoskeleton user images and feedback using Qwen2-VL model.
    
    Args:
        image_path: Path to the input image
        user_feedback: Text feedback from the user
        model_name: Qwen model name to use
        
    Returns:
        Dictionary with task, assistance_level_percent, and reasoning
    """
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    
    # Load and validate image
    if not Path(image_path).exists():
        raise ValueError(f"Image file '{image_path}' not found")
    
    image = Image.open(image_path).convert('RGB')
    print(f"Loaded image: {image_path}")
    
    # Load the prompt from const.py
    try:
        base_prompt = load_prompt_from_const()
        print("Loaded prompt from const.py")
    except Exception as e:
        print(f"Warning: Could not load prompt from const.py: {e}")
        # Fallback to a simple prompt
        base_prompt = """You are a multimodal assistant for an exoskeleton system. Your job is to analyze both an image showing the user's environment or movement context, and textual feedback from the user. 

Your objectives are:

1. Identify the locomotion task the user is performing. Available tasks:
   - Walking uphill
   - Walking downhill
   - Running
   - Walking up stairs
   - Walking down stairs
   - Walking on flat ground

2. Estimate the assistance level (0–10%) based on the user's feedback. 
   - Feedback may be explicit ("give me more help") or implicit ("ow", "too much", "I'm fine").
   - Ensure the assistance level is reasonable and interprets the user's intent.

### Output Format
Return a JSON object with the following fields:

{
  "task": "<one of: uphill, downhill, run, stairs_up, stairs_down, flat>",
  "assistance_level_percent": <float between 0.0 and 10.0>,
  "reasoning": "<brief explanation describing why this decision was made>"
}"""
    
    # Create the final prompt with user feedback
    prompt = f"""{base_prompt}

User Feedback: "{user_feedback}"

You must respond with ONLY a valid JSON object. No additional text, explanations, or formatting."""

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    # Process inputs
    print("Processing image and text...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)

    # Generate response
    print("Generating analysis...")
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=200,
        do_sample=False,  # Use greedy decoding for consistent JSON
        temperature=0.1,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    
    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Extract and parse JSON
    response = output_text[0].strip()
    print(f"Raw response: {response}")
    
    # Find JSON in response
    json_start = response.find('{')
    if json_start != -1:
        response = response[json_start:]
    
    # Find the end of the JSON object
    brace_count = 0
    json_end = -1
    for i, char in enumerate(response):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break
    
    if json_end > 0:
        json_str = response[:json_end]
        try:
            result = json.loads(json_str)
            # Validate required fields
            if all(key in result for key in ['task', 'assistance_level_percent', 'reasoning']):
                return result
            else:
                raise ValueError("Missing required fields in JSON")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error: {e}")
            raise e
    else:
        raise ValueError("No valid JSON found in response")


def main():
    """Main function to run the exoskeleton analyzer."""
    parser = argparse.ArgumentParser(description="Analyze exoskeleton user images and feedback")
    parser.add_argument("--image", "-i", required=True, help="Path to the input image")
    parser.add_argument("--feedback", "-f", required=True, help="User feedback text")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2-VL-2B-Instruct", 
                       help="Qwen model name (default: Qwen/Qwen2-VL-2B-Instruct)")
    parser.add_argument("--output", "-o", help="Output JSON file path (optional)")
    
    args = parser.parse_args()
    
    try:
        # Analyze the scene
        result = analyze_exoskeleton_scene(args.image, args.feedback, args.model)
        
        # Display results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS:")
        print("="*50)
        print(json.dumps(result, indent=2))
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        print(f"\nError: Failed to analyze image and feedback")
        print(f"Details: {e}")
        return 1


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python test.py --image path/to/image.jpg --feedback 'Need more help'")
        print("python test.py -i image.jpg -f 'Too much assistance' -o results.json")
        print("python test.py -i stairs.jpg -f 'Ow, that hurts' -m 'Qwen/Qwen2-VL-2B-Instruct'")
    else:
        main()
