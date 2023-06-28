import os
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig
import streamlit as st


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


def generate_response(user_input):
    generator = generate(llm, generation_config, user_input.strip())
    response = ""
    for word in generator:
        response += word
    return response


if __name__ == "__main__":
    config = AutoConfig.from_pretrained(
        "teknium/Replit-v2-CodeInstruct-3B", context_length=2048
    )
    llm = AutoModelForCausalLM.from_pretrained(
        os.path.abspath("models/replit-v2-codeinstruct-3b.q4_1.bin"),
        model_type="replit",
        config=config,
    )

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

    user_prefix = "[user]: "
    assistant_prefix = "[assistant]: "

    st.title("Chat with Assistant")
    user_input = st.text_input(user_prefix)
    response = generate_response(user_input)
    st.text_area(assistant_prefix, value=response, height=200)
