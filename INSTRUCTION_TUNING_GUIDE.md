# Instruction Tuning Guide for Exoskeleton Model

This guide explains how to perform instruction tuning on the Qwen model for the exoskeleton assistance task.

## Overview

Instruction tuning fine-tunes the Qwen2-VL model specifically for:
- Visual terrain assessment (gentle, moderate, steep)
- Initial assistance level prediction based on terrain
- Delta prediction from previous assistance levels
- Handling vague feedback and pain expressions

## Files Created

1. **`instruction_tuning.py`** - Main training script
2. **`create_dummy_images.py`** - Creates dummy training images
3. **`evaluate_tuned_model.py`** - Evaluates the tuned model
4. **`INSTRUCTION_TUNING_GUIDE.md`** - This guide

## Step-by-Step Process

### 1. Create Dummy Images

First, create the dummy images for training:

```bash
python create_dummy_images.py
```

This creates:
- `data/flat_ground.jpg` - Flat terrain
- `data/stairs_up.jpg` - Stairs going up
- `data/stairs_down.jpg` - Stairs going down  
- `data/uphill.jpg` - Moderate uphill
- `data/uphill_steep.jpg` - Steep uphill

### 2. Generate Training Data

Create the training dataset:

```bash
python instruction_tuning.py --create-data
```

This generates `training_data.json` with training examples including:
- Image paths
- Prompts with placeholders filled
- Expected JSON outputs

### 3. Train the Model

Run instruction tuning:

```bash
python instruction_tuning.py --epochs 3 --batch-size 1
```

Parameters:
- `--epochs` - Number of training epochs (default: 3)
- `--batch-size` - Training batch size (default: 1)
- `--output-dir` - Where to save tuned model (default: ./tuned_model)

### 4. Evaluate the Tuned Model

Test the tuned model:

```bash
# Single test case
python evaluate_tuned_model.py --model-path ./tuned_model --single-test --image data/flat_ground.jpg --feedback "Need help" --previous 5.0

# Full test set (if you have test_data.json)
python evaluate_tuned_model.py --model-path ./tuned_model --test-data test_data.json
```

## Training Data Structure

The training data includes scenarios for:

### Terrain Types
- **Flat ground**: gentle terrain, 3% initial assistance
- **Moderate stairs/incline**: 6% initial assistance  
- **Steep terrain**: 9% initial assistance

### Feedback Types
- **Explicit requests**: "Need more help", "Less assistance please"
- **Pain expressions**: "Ow", "Ouch", "Ugh"
- **Vague feedback**: "Hmm", "Too much"

### Expected Behaviors
- Positive deltas for requests for more help
- Negative deltas for requests for less help
- Larger deltas for pain expressions on steep terrain
- Terrain-aware initial assistance levels

## Customizing Training

### Add More Training Data

Edit `instruction_tuning.py` and add more scenarios to the `scenarios` list:

```python
{
    "image_path": "data/your_image.jpg",
    "user_feedback": "Your feedback",
    "previous_assistance": 5.0,
    "expected_output": {
        "task": "uphill",
        "terrain_difficulty": "moderate", 
        "initial_assistance": 6.0,
        "assistance_delta": 2.0,
        "reasoning": "Your reasoning"
    }
}
```

### Adjust Training Parameters

Modify training arguments in `instruction_tuning.py`:

```python
training_args = TrainingArguments(
    learning_rate=5e-5,  # Adjust learning rate
    num_train_epochs=5,  # More epochs
    per_device_train_batch_size=2,  # Larger batch size
    warmup_steps=200,    # More warmup
)
```

### Create Test Data

Create `test_data.json` with the same structure as training data for evaluation:

```json
[
    {
        "image_path": "data/test_image.jpg",
        "user_feedback": "Test feedback",
        "previous_assistance": 5.0,
        "expected_output": {
            "task": "flat",
            "terrain_difficulty": "gentle",
            "initial_assistance": 3.0,
            "assistance_delta": 1.0,
            "reasoning": "Test reasoning"
        }
    }
]
```

## Using the Tuned Model

After training, use the tuned model in your dialogue system:

```python
# In dialogue_system.py, change the model path:
dialogue_system = ExoskeletonDialogueSystem("./tuned_model")
```

## Expected Improvements

After instruction tuning, the model should:

1. **Better terrain assessment** - More accurate gentle/moderate/steep classification
2. **Consistent initial assistance** - Proper 3%/6%/9% based on terrain
3. **Context-aware deltas** - Larger deltas for pain on steep terrain
4. **Robust JSON output** - More reliable JSON formatting
5. **Better reasoning** - More detailed explanations

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 1`
- Use gradient accumulation: increase `gradient_accumulation_steps`
- Use smaller model: `Qwen/Qwen2-VL-1.5B-Instruct`

### Poor Performance
- Increase training epochs: `--epochs 5`
- Add more diverse training data
- Adjust learning rate
- Check training data quality

### JSON Parsing Errors
- Ensure training data has valid JSON outputs
- Add more examples with proper JSON formatting
- Increase `max_new_tokens` in generation

## Next Steps

1. **Collect real data** - Replace dummy images with real exoskeleton scenarios
2. **Expand training set** - Add more diverse feedback types and terrain
3. **Fine-tune parameters** - Optimize learning rate, epochs, batch size
4. **Evaluate thoroughly** - Test on held-out data
5. **Deploy** - Integrate tuned model into production system

## Hardware Requirements

- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for model and data
- **Training time**: 1-3 hours depending on hardware and data size

