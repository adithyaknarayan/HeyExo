import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import sys
from datetime import datetime

class TunedModelEvaluator:
    """Evaluator for the tuned exoskeleton model."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        print(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the tuned model and processor."""
        print(f"Loading tuned model from: {self.model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)
        print("Tuned model loaded successfully!")
    
    def _load_prompt_from_const(self):
        """Load the prompt from const.py file."""
        const_path = Path(__file__).parent / "const.py"
        if not const_path.exists():
            raise FileNotFoundError("const.py file not found")
        
        with open(const_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the prompt content
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
    
    def evaluate_single(self, image_path, user_feedback, previous_assistance):
        """Evaluate a single test case."""
        if not Path(image_path).exists():
            raise ValueError(f"Image file '{image_path}' not found")
        
        image = Image.open(image_path).convert('RGB')
        
        # Load the prompt from const.py
        base_prompt = self._load_prompt_from_const()
        
        # Replace placeholders in the prompt
        prompt = base_prompt.replace("{USER_FEEDBACK}", user_feedback)
        prompt = prompt.replace("{PREVIOUS_ASSISTANCE}", str(previous_assistance))
        prompt += "\n\nYou must respond with ONLY a valid JSON object. No additional text, explanations, or formatting."
        
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
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=200,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Extract and parse JSON
        response = output_text[0].strip()
        
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
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {response}")
                return None
        else:
            print("No valid JSON found in response")
            return None
    
    def evaluate_test_set(self, test_data_file):
        """Evaluate on a test dataset."""
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        print(f"\nEvaluating on {len(test_data)} test cases...")
        print("="*60)
        
        for i, test_case in enumerate(test_data):
            print(f"\nTest Case {i+1}:")
            print(f"Image: {test_case['image_path']}")
            print(f"Feedback: '{test_case['user_feedback']}'")
            print(f"Previous Assistance: {test_case['previous_assistance']}%")
            
            # Get prediction
            prediction = self.evaluate_single(
                test_case['image_path'],
                test_case['user_feedback'],
                test_case['previous_assistance']
            )
            
            if prediction is None:
                print("❌ Failed to get valid prediction")
                continue
            
            # Display prediction
            print(f"\nPrediction:")
            print(f"  Task: {prediction.get('task', 'N/A')}")
            print(f"  Terrain Difficulty: {prediction.get('terrain_difficulty', 'N/A')}")
            print(f"  Initial Assistance: {prediction.get('initial_assistance', 'N/A')}%")
            print(f"  Delta: {prediction.get('assistance_delta', 'N/A')}")
            print(f"  Reasoning: {prediction.get('reasoning', 'N/A')}")
            
            # Compare with expected
            expected = test_case['expected_output']
            print(f"\nExpected:")
            print(f"  Task: {expected.get('task', 'N/A')}")
            print(f"  Terrain Difficulty: {expected.get('terrain_difficulty', 'N/A')}")
            print(f"  Initial Assistance: {expected.get('initial_assistance', 'N/A')}%")
            print(f"  Delta: {expected.get('assistance_delta', 'N/A')}")
            
            # Check accuracy
            task_correct = prediction.get('task') == expected.get('task')
            terrain_correct = prediction.get('terrain_difficulty') == expected.get('terrain_difficulty')
            delta_close = abs(prediction.get('assistance_delta', 0) - expected.get('assistance_delta', 0)) <= 1.0
            
            if task_correct and terrain_correct and delta_close:
                print("✅ Correct prediction!")
                correct_predictions += 1
            else:
                print("❌ Incorrect prediction")
                if not task_correct:
                    print(f"   Task mismatch: {prediction.get('task')} vs {expected.get('task')}")
                if not terrain_correct:
                    print(f"   Terrain mismatch: {prediction.get('terrain_difficulty')} vs {expected.get('terrain_difficulty')}")
                if not delta_close:
                    print(f"   Delta mismatch: {prediction.get('assistance_delta')} vs {expected.get('assistance_delta')}")
            
            total_predictions += 1
            results.append({
                'test_case': test_case,
                'prediction': prediction,
                'correct': task_correct and terrain_correct and delta_close
            })
        
        # Summary
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n" + "="*60)
        print(f"EVALUATION SUMMARY")
        print(f"="*60)
        print(f"Total test cases: {len(test_data)}")
        print(f"Valid predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return results, accuracy

def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Tuned Exoskeleton Model")
    parser.add_argument("--model-path", "-m", required=True,
                       help="Path to the tuned model")
    parser.add_argument("--test-data", "-t", default="test_data.json",
                       help="Test data file")
    parser.add_argument("--single-test", action="store_true",
                       help="Run single test case")
    parser.add_argument("--image", "-i", help="Image path for single test")
    parser.add_argument("--feedback", "-f", help="User feedback for single test")
    parser.add_argument("--previous", "-p", type=float, default=5.0,
                       help="Previous assistance level for single test")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = TunedModelEvaluator(args.model_path)
        
        if args.single_test:
            if not args.image or not args.feedback:
                print("For single test, provide --image and --feedback")
                return 1
            
            result = evaluator.evaluate_single(args.image, args.feedback, args.previous)
            if result:
                print("\nResult:")
                print(json.dumps(result, indent=2))
            else:
                print("Failed to get result")
                return 1
        else:
            # Evaluate on test set
            if not Path(args.test_data).exists():
                print(f"Test data file {args.test_data} not found.")
                print("Create test data or use --single-test for individual testing.")
                return 1
            
            results, accuracy = evaluator.evaluate_test_set(args.test_data)
            print(f"\nFinal Accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()

