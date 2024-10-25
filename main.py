from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import re


# Load the Hugging Face token from token.txt
with open("token.txt", "r") as token_file:
    hf_token = token_file.read().strip()

# Set up LLaMA model and tokenizer using Hugging Face token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-3.2-1B"

tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to(device)

def read_random_events(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def parse_events(lines):
    events = []
    current_event = []
    for line in lines:
        if line.strip().startswith("#"):
            if current_event:
                events.append(current_event)
                current_event = []
        current_event.append(line.strip())
    if current_event:
        events.append(current_event)
    return events

def llama_generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=500)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def convert_event_with_llama(event_lines):
    event_str = "\n".join(event_lines)
    prompt = f"Convert the following event according to the given triggers and effects, adapting it as closely as possible:\n\n{event_str}\n\nAdapted Event:\n"
    generated_event = llama_generate(prompt)
    return generated_event

def process_and_save_events(input_file, output_file):
    lines = read_random_events(input_file)
    events = parse_events(lines)

    adapted_events = []
    for event in events:
        adapted_event = convert_event_with_llama(event)
        adapted_events.append(adapted_event)

    with open(output_file, 'w') as file:
        for event in adapted_events:
            file.write(event + "\n\n")

# File paths
input_file = 'RandomEvents.txt'
output_file = 'AdaptedEvents.txt'

# Process events and save
process_and_save_events(input_file, output_file)
