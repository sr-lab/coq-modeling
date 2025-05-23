import argparse
import threading
import concurrent.futures
import json
from tactic_gen.train_decoder import (get_datasets, init_valid_files)
from util.train_utils import load_config
from data_management.jsonl_utils import ExampleDB
from pathlib import Path

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
    The tactic ([TACTIC]) is the tactic that you have to get to after reasoning about the state, script, proofs and premises.
    
    Generate the reasoning a human would have to decide which tactic to apply next.
    During the reasoning, ignore that you already know the tactic in the [TACTIC] field. 

    Your reasoning should end with:
    'Therefore, the next tactic is <TACTIC>.'
    <TACTIC> should be replaced by the tactic that follows the [TACTIC] field in the prompt.

    As an example, if the original prompt is:
    [STATE]
    A: Type
    ∀ (x : A) (l : list A), rev (l ++ [x]) = x :: rev l
    [SCRIPT]
    Lemma rev_snoc_cons A :
    forall (x : A) (l : list A), rev (l ++ [x]) = x :: rev l.
    Proof.
    [PROOFS]
    [PREMISES]
    [TACTIC]
    induction l.

    [REASONING]
    The current state is a lemma about the reverse of a list.
    The current script shows the definition of the lemma and the start of the proof.
    I need to prove that the reverse of a list with an element at the end is the element followed by the reverse of the list without the element.
    I can do this by induction on the list.
    Therefore, the next tactic is "induction l."
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


def process_example(proof):
    original_prompt = proof["prompt"]
    first_next_step = proof["next_steps"][0]

    system_prompt, prompt = generate_prompt(original_prompt, first_next_step)
    cot = call_openai(system_prompt, prompt)
    proof["cot"] = cot
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--db_output", type=str, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to use")
    args = parser.parse_args()


    config = load_config(args.config)
    if "repos_path" in config:
        init_valid_files(config["repos_path"])
    train_dataset, val_dataset = get_datasets(config)

    db = ExampleDB.create(Path(args.db_output))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for i in range(args.num_examples):
            proof = train_dataset[i]
            futures.append(executor.submit(process_example, proof))
        
        examples = []
        for future in concurrent.futures.as_completed(futures):
            examples.append(json.dumps(future.result()))
            if len(examples) > 100:
                db.insert_examples(examples)
                examples = []
            
        db.insert_examples(examples)
