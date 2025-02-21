import torch
from transformers import MarianMTModel, MarianTokenizer
import gradio as gr

# Define source and target languages
source_lang = "en"
target_lang = "fr"
model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

# Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate text
def translate_text(text):
    if not text:
        return "Please enter some text."
    
    # Tokenize input text
    input_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation
    with torch.no_grad():
        translated_tokens = model.generate(**input_tokens)

    # Decode output
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Define Gradio interface
interface = gr.Interface(
    fn=translate_text,
    inputs="text",
    outputs="text",
    title="Real-Time Language Translator",
    description="Enter text in English and get the translated text in French."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()