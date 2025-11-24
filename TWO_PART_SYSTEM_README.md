# Two-Part Prompt System with Separate Files Implementation

## Overview

The exoskeleton dialogue system has been refactored to use a two-part prompt system with separate files that separates task identification from assistance level adjustment. This approach provides better modularity, clearer reasoning, improved performance, and easier maintenance.

## System Architecture

### Part 1: Task Identification
- **Purpose**: Analyze the image to identify the locomotion task and assess terrain difficulty
- **Input**: Image only
- **Output**: JSON with task, terrain_difficulty, and confidence score
- **File**: `const_task.py` containing `TASK_IDENTIFICATION_PROMPT`

### Part 2: Assistance Adjustment  
- **Purpose**: Determine assistance level adjustment based on identified task and user feedback
- **Input**: Text-only (task info + user feedback)
- **Output**: JSON with assistance calculations and detailed reasoning
- **File**: `const_assistance.py` containing `ASSISTANCE_ADJUSTMENT_PROMPT`
- **Dynamic Injection**: The identified task is dynamically injected into the assistance prompt

## Key Benefits

1. **Separation of Concerns**: Visual analysis is isolated from assistance logic
2. **Chain of Thought**: The second prompt includes explicit reasoning steps
3. **Confidence Scoring**: Task identification includes confidence metrics
4. **Modularity**: Each part can be optimized independently
5. **Debugging**: Easier to identify issues in specific components
6. **File Organization**: Prompts are stored in separate, focused files
7. **Dynamic Injection**: Task information is dynamically injected into assistance prompt
8. **Maintainability**: Easier to update individual prompts without affecting others

## File Changes

### New Files Created
- **`const_task.py`**: Contains `TASK_IDENTIFICATION_PROMPT` for visual analysis
- **`const_assistance.py`**: Contains `ASSISTANCE_ADJUSTMENT_PROMPT` with chain of thought reasoning

### Files Removed
- **`const.py`**: Replaced with separate, focused files

### Files Modified
- **`dialogue_system.py`**: 
  - Added `identify_task()` method for Part 1
  - Updated `analyze_feedback()` to use two-step process with dynamic injection
  - Modified prompt loading to use separate files (`_load_task_prompt()`, `_load_assistance_prompt()`)
  - Enhanced result display to show confidence scores

- **`instruction_tuning.py`**: 
  - Updated to work with separate prompt files
  - Modified training data creation for assistance adjustment prompt

- **`test_two_part_system.py`**: 
  - Updated to demonstrate separate file loading and dynamic injection

## Usage Example

```python
# Initialize system
dialogue_system = ExoskeletonDialogueSystem()
dialogue_system.set_image("path/to/image.jpg")

# Step 1: Identify task (automatic in analyze_feedback)
# Loads prompt from const_task.py
task_result = dialogue_system.identify_task()
print(f"Task: {task_result['task']} (confidence: {task_result['confidence']:.2f})")

# Step 2: Full analysis with user feedback
# Loads prompt from const_assistance.py and dynamically injects task info
result = dialogue_system.analyze_feedback("Need more help")
print(f"Final assistance: {result['final_assistance_level']}%")
```

## Dynamic Injection Process

1. **Task Identification**: System loads `const_task.py` and analyzes image
2. **Task Extraction**: Extracts task, terrain_difficulty, and confidence
3. **Dynamic Injection**: Loads `const_assistance.py` and replaces placeholders:
   - `{IDENTIFIED_TASK}` → actual task (e.g., "stairs_up")
   - `{TERRAIN_DIFFICULTY}` → terrain assessment (e.g., "moderate")
   - `{PREVIOUS_ASSISTANCE}` → previous assistance level (e.g., "4.0")
   - `{USER_FEEDBACK}` → user input (e.g., "Need more help")
4. **Assistance Calculation**: System processes the injected prompt, returns delta, and calculates final assistance level in code

## Testing

Run the test script to verify the system works correctly:

```bash
python test_two_part_system.py
```

## Output Format

### Task Identification Output
```json
{
  "task": "stairs_up",
  "terrain_difficulty": "moderate", 
  "confidence": 0.85
}
```

### Assistance Adjustment Output
```json
{
  "initial_assistance": 6.0,
  "assistance_delta": 2.5,
  "reasoning": "Moderate stairs detected. User explicitly requesting more help..."
}
```

**Note**: The final assistance level is calculated in code by adding the delta to the previous assistance level.

## Chain of Thought Process

The assistance adjustment prompt follows a structured reasoning process:

1. **Task Analysis**: Review the identified locomotion task
2. **Terrain Assessment**: Consider the terrain difficulty level  
3. **Previous Context**: Account for the previous assistance level
4. **User Feedback Analysis**: Interpret the user's input
5. **Delta Calculation**: Determine appropriate adjustment
6. **Final Calculation**: Compute final assistance level

This structured approach ensures consistent and explainable assistance adjustments.
