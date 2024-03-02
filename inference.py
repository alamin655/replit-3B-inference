import gradio as gr
import os
import spaces
import torch
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig

@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]

def format_prompt(user_prompt: str):
    return f"""### Instruction:
{user_prompt}

### Response:"""

@spaces.GPU
def generate(
    llm: AutoModelForCausalLM,
    generation_config: GenerationConfig,
    user_prompt: str,
):
    """run model inference, will return a Generator if streaming is true"""

    return llm(
        format_prompt(
            user_prompt,
        ),
        **asdict(generation_config),
    )

config = AutoConfig.from_pretrained(
    "teknium/Replit-v2-CodeInstruct-3B", context_length=2048
)

# Check if GPU is available, if not use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm = AutoModelForCausalLM.from_pretrained(
    os.path.abspath("models/replit-v2-codeinstruct-3b.q4_1.bin"),
    model_type="replit",
    config=config,
).to(device)

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    max_new_tokens=512,  # adjust as needed
    seed=42,
    reset=True,  # reset history (cache)
    stream=True,  # streaming per word/token
    threads=int(os.cpu_count() / 6),  # adjust for your CPU
    stop=["<|endoftext|>"],
)

def generate_text(prompt):
    generator = generate(llm, generation_config, prompt.strip())
    response = ""
    for word in generator:
        response += word
    return response

inputs = gr.inputs.Textbox(lines=7, label="Enter your prompt here")
outputs = gr.outputs.Textbox(label="Model response")

title = "Replit Code Instruct inference using GPU (if available)"
description = "This AI can help you with your coding questions. Enter your prompt and get a response from the AI."
examples = [
    ["How do I read a file in Python?"],
    ["What is the difference between a stack and a queue?"],
    ["How do I sort a list of numbers in Python?"],
    ["What is the time complexity of quicksort?"],
    ["How do I implement a binary search in Python?"],
]

gr.Interface(
    generate_text,
    inputs,
    outputs,
    title=title,
    description=description,
    examples=examples,
).launch()
