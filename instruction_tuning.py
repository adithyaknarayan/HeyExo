import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration, 
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset
import sys
from datetime import datetime
import os

class ExoskeletonDataset(Dataset):
    """Dataset for exoskeleton assistance instruction tuning."""
    
    def __init__(self, data_file, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        
        # Load training data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Create conversation format
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
                        "text": item['prompt']
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
        
        # Prepare target (the expected JSON response)
        target_text = json.dumps(item['expected_output'])
        
        # Tokenize target
        target_tokens = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Create labels (same as input_ids for generation)
        labels = target_tokens.input_ids.clone()
        
        return {
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'pixel_values': inputs.pixel_values.squeeze(0),
            'labels': labels.squeeze(0)
        }

class ExoskeletonTrainer:
    """Trainer for exoskeleton assistance model."""
    
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        print(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen model and processor."""
        print(f"Loading model: {self.model_name}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_name)
        print("Model loaded successfully!")
    
    def create_training_data(self, output_file="training_data.json"):
        """Create training data with various scenarios."""
        training_data = []
        
        # Load the assistance adjustment prompt from separate file
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
        
        base_prompt = content[assist_start:assist_end].strip()
        
        # Training scenarios
        scenarios = [
            # Flat ground scenarios
            {
                "image_path": "data/flat_ground.jpg",  # You'll need to create these images
                "user_feedback": "Need more help",
                "previous_assistance": 3.0,
                "expected_output": {
                    "task": "flat",
                    "terrain_difficulty": "gentle",
                    "initial_assistance": 3.0,
                    "assistance_delta": 2.0,
                    "reasoning": "Flat ground detected, user requesting more help, increasing assistance."
                }
            },
            {
                "image_path": "data/flat_ground.jpg",
                "user_feedback": "Too much assistance",
                "previous_assistance": 8.0,
                "expected_output": {
                    "task": "flat",
                    "terrain_difficulty": "gentle",
                    "initial_assistance": 3.0,
                    "assistance_delta": -3.0,
                    "reasoning": "Flat ground detected, user indicating over-assistance, reducing assistance."
                }
            },
            {
                "image_path": "data/flat_ground.jpg",
                "user_feedback": "Ow",
                "previous_assistance": 2.0,
                "expected_output": {
                    "task": "flat",
                    "terrain_difficulty": "gentle",
                    "initial_assistance": 3.0,
                    "assistance_delta": 3.0,
                    "reasoning": "Flat ground detected, user expressing pain, increasing assistance."
                }
            },
            
            # Stairs scenarios
            {
                "image_path": "data/stairs_up.jpg",
                "user_feedback": "This is hard",
                "previous_assistance": 5.0,
                "expected_output": {
                    "task": "stairs_up",
                    "terrain_difficulty": "moderate",
                    "initial_assistance": 6.0,
                    "assistance_delta": 2.5,
                    "reasoning": "Moderate stairs detected, user expressing difficulty, increasing assistance."
                }
            },
            {
                "image_path": "data/stairs_up.jpg",
                "user_feedback": "Ouch",
                "previous_assistance": 4.0,
                "expected_output": {
                    "task": "stairs_up",
                    "terrain_difficulty": "moderate",
                    "initial_assistance": 6.0,
                    "assistance_delta": 4.0,
                    "reasoning": "Moderate stairs detected, user expressing pain, significant assistance increase needed."
                }
            },
            
            # Uphill scenarios
            {
                "image_path": "data/uphill.jpg",
                "user_feedback": "Need a bit more push",
                "previous_assistance": 4.0,
                "expected_output": {
                    "task": "uphill",
                    "terrain_difficulty": "moderate",
                    "initial_assistance": 6.0,
                    "assistance_delta": 2.5,
                    "reasoning": "Moderate incline detected, user requesting more help, increasing assistance."
                }
            },
            {
                "image_path": "data/uphill_steep.jpg",
                "user_feedback": "Ugh",
                "previous_assistance": 6.0,
                "expected_output": {
                    "task": "uphill",
                    "terrain_difficulty": "steep",
                    "initial_assistance": 9.0,
                    "assistance_delta": 3.5,
                    "reasoning": "Steep uphill terrain detected, user expressing difficulty, moderate assistance increase needed."
                }
            },
            
            # Vague feedback scenarios
            {
                "image_path": "data/flat_ground.jpg",
                "user_feedback": "Hmm",
                "previous_assistance": 5.0,
                "expected_output": {
                    "task": "flat",
                    "terrain_difficulty": "gentle",
                    "initial_assistance": 3.0,
                    "assistance_delta": -1.0,
                    "reasoning": "Flat ground detected, vague negative feedback, slight reduction in assistance."
                }
            },
            {
                "image_path": "data/stairs_down.jpg",
                "user_feedback": "Ugh",
                "previous_assistance": 3.0,
                "expected_output": {
                    "task": "stairs_down",
                    "terrain_difficulty": "moderate",
                    "initial_assistance": 6.0,
                    "assistance_delta": 2.5,
                    "reasoning": "Moderate stairs down detected, user expressing difficulty, moderate assistance increase."
                }
            }
        ]
        
        # Create training examples
        for scenario in scenarios:
            # Replace placeholders in prompt
            prompt = base_prompt.replace("{IDENTIFIED_TASK}", scenario["expected_output"]["task"])
            prompt = prompt.replace("{TERRAIN_DIFFICULTY}", scenario["expected_output"]["terrain_difficulty"])
            prompt = prompt.replace("{PREVIOUS_ASSISTANCE}", str(scenario["previous_assistance"]))
            prompt = prompt.replace("{USER_FEEDBACK}", scenario["user_feedback"])
            prompt += "\n\nYou must respond with ONLY a valid JSON object. No additional text, explanations, or formatting."
            
            training_data.append({
                "image_path": scenario["image_path"],
                "prompt": prompt,
                "expected_output": scenario["expected_output"]
            })
        
        # Save training data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created {len(training_data)} training examples in {output_file}")
        return training_data
    
    def train(self, training_data_file, output_dir="./tuned_model", num_epochs=3, batch_size=1):
        """Train the model on exoskeleton assistance task."""
        
        # Create dataset
        dataset = ExoskeletonDataset(training_data_file, self.processor)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer

def main():
    """Main function for instruction tuning."""
    parser = argparse.ArgumentParser(description="Instruction Tuning for Exoskeleton Model")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2-VL-2B-Instruct", 
                       help="Base model name")
    parser.add_argument("--data-file", "-d", default="training_data.json",
                       help="Training data file")
    parser.add_argument("--output-dir", "-o", default="./tuned_model",
                       help="Output directory for tuned model")
    parser.add_argument("--epochs", "-e", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--create-data", action="store_true",
                       help="Create training data")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ExoskeletonTrainer(args.model)
        
        # Create training data if requested
        if args.create_data:
            trainer.create_training_data(args.data_file)
            print("Training data created. Please add your images to the data/ directory.")
            print("Then run without --create-data to start training.")
            return
        
        # Check if training data exists
        if not Path(args.data_file).exists():
            print(f"Training data file {args.data_file} not found.")
            print("Run with --create-data to generate training data first.")
            return
        
        # Train model
        trainer.train(
            training_data_file=args.data_file,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()

