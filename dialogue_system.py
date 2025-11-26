import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import sys
from datetime import datetime
import os
import base64
from io import BytesIO
import re

# Try to import OpenAI and dotenv
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class ExoskeletonDialogueSystem:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """Initialize the dialogue system with a Qwen model or OpenAI model."""
        self.model_name = model_name
        self.is_openai = "gpt-" in model_name.lower() or "o1-" in model_name.lower()
        
        self.baseline_file = Path("baseline_preference.txt")
        self.previous_assistance_file = Path("previous_assistance.txt")
        self.image = None
        self.model = None
        self.processor = None
        self.openai_client = None
        
        # Determine feature support based on model name
        is_o1 = "o1-" in model_name.lower()
        self.openai_supports_temperature = not is_o1
        self.openai_supports_response_format = not is_o1
        
        self.baseline_assistance = None
        self.previous_assistance = None
        
        if self.is_openai:
            if not HAS_OPENAI:
                raise ImportError("OpenAI package is required for GPT models. Please install it with: pip install openai python-dotenv")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in a .env file.")
            self.openai_client = OpenAI(api_key=api_key)
            print(f"Initialized OpenAI client with model: {self.model_name}")
        else:
            # Check for MPS (Apple Silicon) or CUDA
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            print(f"Using device: {self.device}")
            self._load_model()
            
        self._load_baseline_preference()
        self._load_previous_assistance()
    
    def _load_model(self):
        """Load the Qwen model and processor."""
        print(f"Loading model: {self.model_name}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_name)
        print("Model loaded successfully!")
    
    def _load_baseline_preference(self):
        """Load or establish baseline assistance preference."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                self.baseline_assistance = float(f.read().strip())
            print(f"Loaded baseline preference: {self.baseline_assistance}%")
        else:
            self._establish_baseline()
    
    def _establish_baseline(self):
        """Establish baseline assistance preference through user interaction."""
        print("\n" + "="*60)
        print("BASELINE ASSISTANCE PREFERENCE SETUP")
        print("="*60)
        print("To provide personalized assistance, I need to understand your baseline preference.")
        print("This will be your default assistance level that I'll adjust up or down based on your feedback.")
        print("="*60)
        
        while True:
            try:
                response = input("\nWhat assistance level do you generally prefer? (0-20%): ").strip()
                baseline = float(response)
                
                if 0.0 <= baseline <= 20.0:
                    self.baseline_assistance = baseline
                    # Save baseline preference
                    with open(self.baseline_file, 'w', encoding='utf-8') as f:
                        f.write(str(baseline))
                    print(f"Baseline preference set to {baseline}%")
                    break
                else:
                    print("Please enter a value between 0.0 and 20.0")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nUsing default baseline of 5.0%")
                self.baseline_assistance = 5.0
                with open(self.baseline_file, 'w', encoding='utf-8') as f:
                    f.write("5.0")
                break
    
    def _load_previous_assistance(self):
        """Load the previous assistance level from file."""
        if self.previous_assistance_file.exists():
            with open(self.previous_assistance_file, 'r', encoding='utf-8') as f:
                self.previous_assistance = float(f.read().strip())
            print(f"Loaded previous assistance: {self.previous_assistance}%")
        else:
            # Use baseline as the initial previous assistance
            self.previous_assistance = self.baseline_assistance
            print(f"No previous assistance found, using baseline: {self.previous_assistance}%")
    
    def _save_previous_assistance(self, assistance_level: float):
        """Save the current assistance level as the previous for next time."""
        with open(self.previous_assistance_file, 'w', encoding='utf-8') as f:
            f.write(str(assistance_level))
        self.previous_assistance = assistance_level
    
    def _load_task_prompt(self):
        """Load the task identification prompt from const_task.py file."""
        task_const_path = Path(__file__).parent / "const_task.py"
        if not task_const_path.exists():
            raise FileNotFoundError("const_task.py file not found")
        
        with open(task_const_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the task identification prompt
        task_start = content.find('TASK_IDENTIFICATION_PROMPT = """')
        if task_start == -1:
            raise ValueError("Could not find TASK_IDENTIFICATION_PROMPT in const_task.py")
        
        task_start += len('TASK_IDENTIFICATION_PROMPT = """')
        task_end = content.find('"""', task_start)
        if task_end == -1:
            raise ValueError("Could not find end of TASK_IDENTIFICATION_PROMPT")
        
        task_prompt = content[task_start:task_end].strip()
        return task_prompt
    
    def _load_assistance_prompt(self):
        """Load the assistance adjustment prompt from const_assistance.py file."""
        assist_const_path = Path(__file__).parent / "const_assistance.py"
        if not assist_const_path.exists():
            raise FileNotFoundError("const_assistance.py file not found")
        
        with open(assist_const_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the assistance adjustment prompt
        assist_start = content.find('ASSISTANCE_ADJUSTMENT_PROMPT = """')
        if assist_start == -1:
            raise ValueError("Could not find ASSISTANCE_ADJUSTMENT_PROMPT in const_assistance.py")
        
        assist_start += len('ASSISTANCE_ADJUSTMENT_PROMPT = """')
        assist_end = content.find('"""', assist_start)
        if assist_end == -1:
            raise ValueError("Could not find end of ASSISTANCE_ADJUSTMENT_PROMPT")
        
        assist_prompt = content[assist_start:assist_end].strip()
        return assist_prompt
    
    def set_image(self, image_path: str):
        """Set the image for the dialogue session."""
        if not Path(image_path).exists():
            raise ValueError(f"Image file '{image_path}' not found")
        
        self.image = Image.open(image_path).convert('RGB')
        print(f"Image loaded: {image_path}")
        print(f"Image size: {self.image.size}")
    
    def _encode_image_to_base64(self):
        """Encode the current PIL image to base64 for OpenAI API."""
        if self.image is None:
            return None
        buffered = BytesIO()
        self.image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _message_content_to_text(self, content) -> str:
        """Convert OpenAI message content (string, list, or SDK object) to plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if item is None:
                    continue
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item.get("text") or "")
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _create_openai_chat_completion(
        self,
        messages,
        max_completion_tokens=300,
        temperature=0.1,
        response_format=None,
        **extra_kwargs
    ):
        """
        Create an OpenAI chat completion, falling back to the default temperature when the
        selected model rejects custom temperature values.
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI client is not initialized.")

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            **extra_kwargs
        }

        include_temperature = temperature is not None and self.openai_supports_temperature
        include_response_format = response_format is not None and self.openai_supports_response_format
        if include_temperature:
            payload["temperature"] = temperature
        if include_response_format:
            payload["response_format"] = response_format

        while True:
            try:
                return self.openai_client.chat.completions.create(**payload)
            except Exception as e:
                error_message = getattr(e, "message", str(e))
                retry = False

                if include_temperature and "temperature" in error_message and "unsupported" in error_message:
                    print("Temperature not supported by this OpenAI model, retrying with default temperature.")
                    self.openai_supports_temperature = False
                    include_temperature = False
                    payload.pop("temperature", None)
                    retry = True

                if include_response_format and "response_format" in error_message and "unsupported" in error_message:
                    print("Response format not supported by this OpenAI model, retrying without it.")
                    self.openai_supports_response_format = False
                    include_response_format = False
                    payload.pop("response_format", None)
                    retry = True

                if retry:
                    continue
                raise

    def _extract_json_from_response(self, response_str: str) -> dict:
        """
        Extract and parse JSON from a response string using multiple strategies.
        This handles clean JSON, Markdown code blocks, and embedded JSON.
        """
        response_str = response_str.strip()
        
        # Strategy 1: Attempt to parse the raw response directly
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from Markdown code blocks (```json ... ```)
        json_code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", response_str, re.DOTALL)
        if json_code_block:
            try:
                return json.loads(json_code_block.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find the largest substring between { and } that forms valid JSON
        # This handles cases where there's text before/after the JSON
        try:
            start_indices = [i for i, char in enumerate(response_str) if char == '{']
            end_indices = [i for i, char in enumerate(response_str) if char == '}']
            
            # Try to match every starting { with every subsequent ending }
            # Iterate from largest range to smallest to find the main object
            for start in start_indices:
                for end in reversed(end_indices):
                    if end > start:
                        possible_json = response_str[start : end + 1]
                        try:
                            return json.loads(possible_json)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass
            
        # Strategy 4: If all else fails, use a fallback brace counter (legacy support)
        # This is similar to what was there before but wrapped as a last resort
        json_start = response_str.find('{')
        if json_start != -1:
            snippet = response_str[json_start:]
            brace_count = 0
            json_end = -1
            for i, char in enumerate(snippet):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > 0:
                try:
                    return json.loads(snippet[:json_end])
                except json.JSONDecodeError:
                    pass

        # If we reach here, extraction failed
        print(f"FAILED TO EXTRACT JSON FROM: {response_str!r}")
        raise ValueError("No valid JSON found in response")

    def identify_task(self) -> dict:
        """Identify the locomotion task from the image."""
        if self.image is None:
            raise ValueError("No image set. Please set an image first.")
        
        # Load the task identification prompt
        task_prompt = self._load_task_prompt()
        
        # Add instruction for JSON output
        prompt = task_prompt + "\n\nYou must respond with ONLY a valid JSON object. No additional text, explanations, or formatting."
        
        if self.is_openai:
            # OpenAI implementation
            base64_image = self._encode_image_to_base64()
            try:
                response = self._create_openai_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_completion_tokens=300,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                message = response.choices[0].message
                parsed_payload = getattr(message, "parsed", None)
                if parsed_payload:
                    return parsed_payload # If already parsed, return it directly
                
                response_str = self._message_content_to_text(message.content).strip()
            except Exception as e:
                print(f"OpenAI API Error: {e}")
                raise e
        else:
            # Qwen implementation
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": self.image
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
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=100,
                    do_sample=False,  # Use greedy decoding for consistent JSON
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            response_str = output_text[0].strip()
        
        # Use robust extraction
        result = self._extract_json_from_response(response_str)
        
        # Validate required fields
        if all(key in result for key in ['task', 'terrain_difficulty', 'confidence']):
            return result
        else:
            raise ValueError(f"Missing required fields in task identification JSON: {result}")
    
    def analyze_feedback(self, user_feedback: str) -> dict:
        """Analyze user feedback using two-part system: task identification + assistance adjustment."""
        if self.image is None:
            raise ValueError("No image set. Please set an image first.")
        
        print("Step 1: Identifying locomotion task from image...")
        # Step 1: Identify the task from the image
        task_result = self.identify_task()
        print(f"Identified task: {task_result['task']} (confidence: {task_result['confidence']:.2f})")
        
        print("Step 2: Determining assistance adjustment...")
        # Step 2: Use the identified task to determine assistance adjustment
        assist_prompt = self._load_assistance_prompt()
        
        # Dynamically inject the identified task and other parameters into the assistance prompt
        prompt = assist_prompt.replace("{IDENTIFIED_TASK}", task_result['task'])
        prompt = prompt.replace("{TERRAIN_DIFFICULTY}", task_result['terrain_difficulty'])
        prompt = prompt.replace("{PREVIOUS_ASSISTANCE}", str(self.previous_assistance))
        prompt = prompt.replace("{USER_FEEDBACK}", user_feedback)
        
        # Add instruction for JSON output
        prompt += "\n\nYou must respond with ONLY a valid JSON object. No additional text, explanations, or formatting."
        
        if self.is_openai:
            # OpenAI Implementation for text-only step
            try:
                response = self._create_openai_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=300,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                message = response.choices[0].message
                parsed_payload = getattr(message, "parsed", None)
                if parsed_payload:
                    assist_result = parsed_payload
                    response_str = None # Signal that we have the result
                else:
                    response_str = self._message_content_to_text(message.content).strip()
                    assist_result = None
            except Exception as e:
                print(f"OpenAI API Error: {e}")
                raise e
        else:
            # Qwen Implementation
            # Prepare messages (text-only for assistance adjustment)
            messages = [
                {
                    "role": "user",
                    "content": [
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
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=300,
                    do_sample=False,  # Use greedy decoding for consistent JSON
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            response_str = output_text[0].strip()
            assist_result = None
        
        # Use robust extraction if not already parsed
        if assist_result is None:
            assist_result = self._extract_json_from_response(response_str)

        # Validate required fields
        if all(key in assist_result for key in ['initial_assistance', 'assistance_delta', 'reasoning']):
            # Calculate final assistance level by adding delta to previous assistance
            delta = float(assist_result['assistance_delta'])
            final_assistance = self.previous_assistance + delta
            # Ensure final assistance level is within bounds
            final_assistance = max(0.0, min(20.0, final_assistance))
            
            # Combine task identification and assistance adjustment results
            result = {
                'task': task_result['task'],
                'terrain_difficulty': task_result['terrain_difficulty'],
                'confidence': task_result['confidence'],
                'initial_assistance': assist_result['initial_assistance'],
                'assistance_delta': assist_result['assistance_delta'],
                'reasoning': assist_result['reasoning'],
                'final_assistance_level': final_assistance
            }
            return result
        else:
            raise ValueError(f"Missing required fields in assistance adjustment JSON: {assist_result}")
    
    def run_dialogue(self):
        """Run the interactive dialogue system."""
        print("\n" + "="*60)
        print("EXOSKELETON DIALOGUE SYSTEM")
        print("="*60)
        print("Type 'quit', 'exit', or 'q' to end the session")
        print("Type 'clear' to reset baseline preference")
        print(f"Current baseline preference: {self.baseline_assistance}%")
        print(f"Previous assistance level: {self.previous_assistance}%")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    if self.baseline_file.exists():
                        self.baseline_file.unlink()
                    if self.previous_assistance_file.exists():
                        self.previous_assistance_file.unlink()
                    print("Baseline preference and previous assistance cleared.")
                    self._establish_baseline()
                    self._load_previous_assistance()
                    continue
                elif not user_input:
                    continue
                
                # Analyze the feedback
                print("Analyzing...")
                result = self.analyze_feedback(user_input)
                
                # Display result
                print("\n" + "-"*50)
                print("ANALYSIS RESULT:")
                print("-"*50)
                print(f"Task: {result['task']} (confidence: {result['confidence']:.2f})")
                print(f"Terrain Difficulty: {result['terrain_difficulty']}")
                print(f"Suggested Initial: {result['initial_assistance']}%")
                print(f"Previous Assistance: {self.previous_assistance}%")
                print(f"Delta: {result['assistance_delta']:+.1f}%")
                print(f"Final Assistance: {result['final_assistance_level']}%")
                print(f"Reasoning: {result['reasoning']}")
                print("-"*50)
                
                # Save the final assistance as the previous for next time
                self._save_previous_assistance(result['final_assistance_level'])
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")


def main():
    """Main function to run the dialogue system."""
    parser = argparse.ArgumentParser(description="Exoskeleton Dialogue System")
    parser.add_argument("--image", "-i", required=True, help="Path to the input image")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2-VL-7B-Instruct", 
                       help="Qwen model name (default: Qwen/Qwen2-VL-7B-Instruct) or OpenAI model (e.g. gpt-4o)")
    
    args = parser.parse_args()
    
    try:
        # Initialize the dialogue system
        dialogue_system = ExoskeletonDialogueSystem(args.model)
        
        # Set the image
        dialogue_system.set_image(args.image)
        
        # Run the dialogue
        dialogue_system.run_dialogue()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
