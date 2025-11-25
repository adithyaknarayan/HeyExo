# Exoskeleton Dialogue System

An interactive system that analyzes visual input and user feedback to adjust exoskeleton assistance levels. The system uses a two-part Chain-of-Thought approach to first identify the locomotion task/terrain from an image, and then determine the appropriate assistance adjustment based on user feedback.

## Setup & Installation

The project includes convenient scripts for installation and execution.

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended) or Apple Silicon Mac (M1/M2/M3)
- 16GB+ RAM recommended for model inference

### Quick Start

1.  **Install Dependencies**
    ```bash
    ./install.sh
    ```
    This will create a virtual environment (`venv`) and install all required packages.

2.  **Run the Application**
    ```bash
    ./run.sh
    ```
    This launches the Streamlit web interface in your default browser.

3.  **Deploy to Server**
    ```bash
    ./deploy.sh
    ```
    Checks for all required files and prepares the environment for deployment.

## Usage

1.  **Upload an Image**: Select an image representing the user's current environment (e.g., stairs, hill, flat ground).
2.  **Chat**: Type feedback like "This is hard", "Need more help", or "Too much push".
3.  **Monitor**: View the system's analysis:
    *   **Task**: What the system sees (e.g., Walking up stairs).
    *   **Terrain**: Difficulty assessment (e.g., Steep).
    *   **Assistance**: How the assistance level changes based on your feedback.

## Architecture

The system uses a two-part prompt architecture:
1.  **Task Identification** (`const_task.py`): Analyzes the image to determine context.
2.  **Assistance Adjustment** (`const_assistance.py`): Uses the identified task + user feedback to calculate specific assistance changes using Chain-of-Thought reasoning.

For more details on the implementation, see [TWO_PART_SYSTEM_README.md](TWO_PART_SYSTEM_README.md).

