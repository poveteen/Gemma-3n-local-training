import gradio as gr

# Global cache to hold model, tokenizer, token, and chat history
# ------------------------------------------------------------------
# Monkey-patch Gradio to skip the buggy OpenAPI schema generation.
# This prevents the TypeError raised inside gradio_client.utils.
import gradio.blocks as _gr_blocks

# Keep original method to fall back if it works
_ORIG_GET_API_INFO = _gr_blocks.Blocks.get_api_info

def _safe_get_api_info(self):
    """Call original get_api_info; if it fails (legacy bug), return an empty dict so frontend defaults work."""
    try:
        return _ORIG_GET_API_INFO(self)
    except Exception:
        # The API schema generation is buggy, return an empty dict to prevent a server crash
        # and allow the frontend to load without the "No API found" error.
        return {}

# Universally apply the patch to fix the persistent API schema bug.
_gr_blocks.Blocks.get_api_info = _safe_get_api_info
# ------------------------------------------------------------------
GLOBAL_CACHE = {}
import pandas as pd
import docx
import pypdf
import os
from huggingface_hub import login
import json
import sys
import io
from contextlib import redirect_stdout

from core.train import run_training_from_app
from core.infer import load_model_for_inference, run_inference_from_app

def save_as_jsonl(data: list[dict], path: str):
    """Saves a list of dictionaries to a JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def generate_instruction_response_pairs(text: str) -> list[dict]:
    """
    A simple function to convert raw text into instruction/response pairs.
    Assumes alternating lines are instruction and response.
    This is a placeholder for more advanced NLP.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        instruction = lines[i]
        response = lines[i+1]
        # A simple heuristic to remove prefixes if they exist
        if instruction.lower().startswith('instruction:'):
            instruction = instruction[len('instruction:'):].strip()
        if response.lower().startswith('response:'):
            response = response[len('response:'):].strip()
        pairs.append({"instruction": instruction, "response": response})
    return pairs



def process_data(file_obj, hf_token, model_name):
    """Handles the data processing and setup when the button is clicked."""
    if file_obj is None:
        return "Error: Please upload a file first."
    if not hf_token:
        return "Error: Please enter your Hugging Face token."

    status_updates = []
    
    try:
        # 0. Ensure default Hugging Face endpoint (avoid cached config overriding base URL)
        os.environ.pop("HF_ENDPOINT", None)
        os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
        # Explicitly force default endpoint in case config file overrides
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"

        # 1. Login to Hugging Face
        status_updates.append("Logging into Hugging Face...")
        login(token=hf_token)
        GLOBAL_CACHE["token"] = hf_token  # cache for later use
        status_updates.append("Successfully logged in.")

        # 2. Check for Gemma terms acceptance (this is a simplified check)
        status_updates.append(f"Please ensure you have accepted the terms for {model_name} on the Hugging Face website.")
        status_updates.append("You will be blocked from downloading if you haven't.")

        # 3. Process the uploaded file
        file_path = file_obj.name
        status_updates.append(f"Processing uploaded file: {os.path.basename(file_path)}...")
        
        # This is a simplified text extraction. We will enhance this.
        if file_path.endswith('.pdf'):
            reader = pypdf.PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages])
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_path.endswith(('.csv', '.xls', '.xlsx')):
            df = pd.read_excel(file_path) if file_path.endswith(('.xls', '.xlsx')) else pd.read_csv(file_path)
            # This assumes a simple two-column structure for now.
            # We will replace this with our advanced NLP extraction later.
            pairs = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
            text = "\n".join([f"Instruction: {p[0]}\nResponse: {p[1]}" for p in pairs])
        else:
            return "Error: Unsupported file type."

        status_updates.append("Successfully extracted text from file.")

        # 4. Convert to JSONL (using our existing logic)
        status_updates.append("Generating instruction/response pairs...")
        instruction_response_pairs = generate_instruction_response_pairs(text)
        
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "train.jsonl")
        save_as_jsonl(instruction_response_pairs, output_path)
        status_updates.append(f"Successfully created training data at {output_path}")
        status_updates.append("\nSetup complete! You can now proceed to the 'Training' tab.")

    except Exception as e:
        return f"An error occurred: {str(e)}"

    return "\n".join(status_updates)


def training_interface(model_name, smoke_test):
    """Interface function to handle training. Can run a full session or a short smoke test."""
    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        try:
            print(f"Starting GPU fine-tuning for {model_name}…\n")

            train_kwargs = {}
            if smoke_test:
                train_kwargs["max_steps"] = 10
                print("--- SMOKE TEST MODE: Training will be limited to 10 steps. ---\n")

            gen = run_training_from_app(
                model_name=model_name,
                train_data_path="data/train.jsonl",
                output_dir="models/gemma-finetuned",
                **train_kwargs,
            )
            for chunk in gen:
                print(chunk)

            # Load freshly saved adapter for inference
            model, tokenizer = load_model_for_inference("models/gemma-finetuned")
            GLOBAL_CACHE["model"] = model
            GLOBAL_CACHE["tokenizer"] = tokenizer
            GLOBAL_CACHE["history"] = []
            print("\nTraining complete! Model cached for chat.")
        except Exception as e:
            print(f"An error occurred during training: {e}")

    return log_stream.getvalue()


def predict(message):
    """Prediction function for the chat interface. Non-streaming. History is stored in chat_state['history']."""
    model = GLOBAL_CACHE.get("model")
    tokenizer = GLOBAL_CACHE.get("tokenizer")
    history = GLOBAL_CACHE.get("history", [])

    if model is None or tokenizer is None:
        history.append((message, "Error: Model not trained or loaded. Please train a model first."))
        return history, ""

    # The history needs to be formatted for the model
    prompt = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
    prompt += f"\nUser: {message}\nAssistant:"

    # Run inference and get the full response as a single string
    response_stream = run_inference_from_app(model, tokenizer, prompt)
    full_response = "".join(list(response_stream))

    # Update history
    history.append((message, full_response))
    GLOBAL_CACHE["history"] = history

    # Return history for the Chatbot and reset the input box
    return history, ""


def push_to_hub_interface(hf_repo_name):
    """Pushes the fine-tuned adapter model to the Hugging Face Hub."""
    token = GLOBAL_CACHE.get("token")
    if not hf_repo_name:
        return "Error: Please provide a repository name."
    if not token:
        return "Error: Hugging Face token not found. Please complete Step 1."

    try:
        # Login to Hugging Face Hub
        login(token=token)

        # Load the trained model from the standard path
        output_dir = "./models/gemma-finetuned"
        if not os.path.exists(output_dir):
            return f"Error: Trained model not found at {output_dir}. Please train a model first."
        
        model, _ = load_model_for_inference(output_dir)
        
        # Push to hub
        model.push_to_hub(hf_repo_name, use_auth_token=True)
        
        hub_url = f"https://huggingface.co/{hf_repo_name}"
        return f"Successfully pushed model to Hub! View it here: {hub_url}"

    except Exception as e:
        return f"An error occurred: {str(e)}"


def main():
    """Main function to create and launch the Gradio web app."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Gemma 3n CPU Trainer") as demo:
        gr.Markdown("# Gemma 3n CPU Fine-Tuning and Inference")
        
        # State objects to hold data across tabs
        chat_state = gr.State(None)
        # Removed hf_token_state

        with gr.Tab("1. Setup & Data Processing"):
            gr.Markdown("## Step 1: Configure Your Project")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload Your Data")
                    file_uploader = gr.File(file_types=[".csv", ".xls", ".xlsx", ".pdf", ".docx"], label="Upload a single CSV, Excel, PDF, or Word file.")
                    
                    gr.Markdown("### Select Model & Provide Token")
                    model_selector = gr.Dropdown([
                        "google/gemma-3-1b-it",
                        "google/gemma-3-2b-it"
                    ], label="Select Gemma Model", value="google/gemma-3-1b-it")
                    hf_token_input = gr.Textbox(label="Hugging Face Token", type="password", placeholder="Enter your Hugging Face token here...")
                    process_button = gr.Button("Process Data & Prepare for Training", variant="primary")

                with gr.Column():
                    gr.Markdown("### Processing Status & Output")
                    status_output = gr.Textbox(label="Status", lines=15, interactive=False, placeholder="Processing status will appear here...")

            process_button.click(
                fn=process_data,
                inputs=[file_uploader, hf_token_input, model_selector],
                outputs=status_output
            )

        with gr.Tab("2. Training"):
            gr.Markdown("## Step 2: Train Your Model")
            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=["google/gemma-3-1b-it", "google/gemma-3-2b-it"],
                    label="Select Base Model",
                    value="google/gemma-3-1b-it",
                )
                start_training_button = gr.Button("Start Fine-Tuning", variant="primary")
            
            smoke_test_checkbox = gr.Checkbox(label="Run smoke test (10 steps)", value=True)
            training_logs = gr.Textbox(label="Training Logs", lines=20, interactive=False, placeholder="Training logs will appear here...")

            # Wire the button to the training function
            start_training_button.click(
                fn=training_interface, 
                inputs=[model_dd, smoke_test_checkbox], 
                outputs=[training_logs]
            )

        with gr.Tab("3. Inference & Export"):
            gr.Markdown("## Step 3: Test and Share Your Model")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Chat with your fine-tuned model")
                    chatbot = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)
                    msg = gr.Textbox(label="Your message", placeholder="Type a message and press Enter...", scale=4)
                    clear = gr.Button("Clear Chat", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("### Export to Hugging Face Hub")
                    hf_repo_name = gr.Textbox(label="New Hub Repo Name", placeholder="e.g., username/gemma-finetuned-sdg")
                    push_to_hub_button = gr.Button("Push to Hub", variant="primary")
                    export_status = gr.Textbox(label="Status", interactive=False)

            # Wire up the chat interface
            msg.submit(predict, [msg], [chatbot, msg])
            clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

            # Wire up the push to hub button
            push_to_hub_button.click(
                fn=push_to_hub_interface,
                inputs=[hf_repo_name],
                outputs=[export_status]
            )

    demo.launch(share=True, show_api=False)


if __name__ == "__main__":
    main()
