import argparse
from tactic_gen.train_decoder import (get_datasets, init_valid_files)
from util.train_utils import load_config


from openai import OpenAI

client = OpenAI()

model = "gpt-4.1-mini"


def generate_prompt(original_prompt: str, first_next_step: str) -> str:
    system_prompt = """You are a helpful assistant that generates a chain of thought (CoT) for
    applying a tactic to an incomplete proof.
    The proof contains a state, a script, similar proofs and premises.
    The state ([STATE]) is the current state of the proof.
    The script ([SCRIPT]) is the sequence of tactics applied to the proof until the current step.
    The proofs ([PROOFS]) are similar proofs.
    The premises ([PREMISES]) are possible auxiliary premises to the current proof.
    The tactic ([TACTIC]) is the next tactic to apply.
    
    Generate a step-by-step reasoning for how we can deduce the next tactic from the state, script, proofs and premises.
    """

    return system_prompt, f"{original_prompt}{first_next_step}"

def call_openai(system_prompt: str, prompt: str) -> str:
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=2000)
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()


    config = load_config(args.config)
    if "repos_path" in config:
        init_valid_files(config["repos_path"])
    train_dataset, val_dataset = get_datasets(config)

    for proof in train_dataset:
        original_prompt = proof["prompt"]
        first_next_step = proof["next_steps"][0]

        system_prompt, prompt = generate_prompt(original_prompt, first_next_step)
        cot = call_openai(system_prompt, prompt)
        print(prompt)
        print(first_next_step)
        print("-------------")
        print(cot)
        break
