# config.py

# --- Model Configuration ---
# You can swap this with other T5 models like 't5-base' or 't5-large'
MODEL_NAME = "t5-small"

# --- Default File Paths ---
# Assumes you have a 'data/' folder for input and 'output/' for results
DEFAULT_INPUT_PATH = "data/Synthetic Medical Notes.csv"
DEFAULT_OUTPUT_PATH = "output/summarized_medical_notes.csv"

# --- Generation Parameters ---
# Parameters for text generation with the model
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 150 # Increased for better summaries
