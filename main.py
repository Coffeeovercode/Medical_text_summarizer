# main.py

import argparse
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import settings from the config file
import config

class MedicalNLP:
    """
    A class to handle summarization and question answering on medical text
    using a T5 model.
    """
    def __init__(self, model_name: str):
        """
        Initializes the tokenizer and model.
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print(f"Model loaded successfully on device: {self.device}")

    def summarize(self, text: str) -> str:
        """
        Generates a summary for a given text.
        """
        input_text = f"summarize: {text}"
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=config.MAX_INPUT_LENGTH,
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            input_ids,
            max_length=config.MAX_OUTPUT_LENGTH,
            num_beams=config.NUM_BEAMS,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def ask_question(self, note: str, question: str) -> str:
        """
        Answers a question based on a given context (note).
        """
        prompt = f"question: {question} context: {note}"
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=config.MAX_INPUT_LENGTH,
            truncation=True
        ).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            max_length=config.MAX_OUTPUT_LENGTH,
            num_beams=config.NUM_BEAMS,
            early_stopping=True
        )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer

def run_summarization(args, processor: MedicalNLP):
    """
    Loads data, runs summarization, and saves the output.
    """
    try:
        print(f"Loading data from: {args.input_file}")
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'.")
        print("Please ensure the file exists and the path in config.py is correct.")
        return

    if 'clinical_note' not in df.columns:
        print("Error: Input CSV must contain a 'clinical_note' column.")
        return

    print("Starting summarization process...")
    df['summary'] = df['clinical_note'].apply(lambda x: processor.summarize(x))
    print("Summarization complete.")

    df.to_csv(args.output_file, index=False)
    print(f"Results saved to: {args.output_file}")
    print("\n--- Sample Summaries ---")
    print(df[['clinical_note', 'summary']].head())
    print("------------------------")


def run_qa(args, processor: MedicalNLP):
    """
    Runs question answering on a single note and question.
    """
    answer = processor.ask_question(note=args.note, question=args.question)
    print(f"\nNote: {args.note}")
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")


def main():
    """
    Main function to parse arguments and run the selected task.
    """
    parser = argparse.ArgumentParser(description="Summarize and ask questions on medical notes.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Summarize command ---
    parser_summarize = subparsers.add_parser("summarize", help="Summarize a CSV file of clinical notes.")
    parser_summarize.add_argument("--input_file", type=str, default=config.DEFAULT_INPUT_PATH, help="Path to the input CSV file.")
    parser_summarize.add_argument("--output_file", type=str, default=config.DEFAULT_OUTPUT_PATH, help="Path to save the output CSV file.")
    parser_summarize.set_defaults(func=run_summarization)

    # --- Q&A command ---
    parser_qa = subparsers.add_parser("qa", help="Ask a question about a single clinical note.")
    parser_qa.add_argument("--note", type=str, required=True, help="The clinical note text.")
    parser_qa.add_argument("--question", type=str, required=True, help="The question to ask.")
    parser_qa.set_defaults(func=run_qa)

    args = parser.parse_args()

    # Initialize the NLP processor
    processor = MedicalNLP(model_name=config.MODEL_NAME)

    # Execute the function associated with the chosen command
    args.func(args, processor)


if __name__ == "__main__":
    main()