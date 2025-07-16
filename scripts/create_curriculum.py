#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import json

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from tactic_gen.lm_example import GeneralFormatter, GeneralFormatterConf

from data_management.dataset_file import DatasetFile
from data_management.sentence_db import SentenceDB
from data_management.jsonl_utils import ExampleDB
from premise_selection.premise_conf import SparseConf, PremiseFilterConf, SparseKind

def analyze_proof_step_complexity(dataset_file: DatasetFile, proof_idx: int, step_idx: int) -> dict:
    """
    Analyze a proof step and return information about its complexity and context.
    This is the foundation for your curriculum sorting.
    """
    proof = dataset_file.proofs[proof_idx]
    step = proof.steps[step_idx]
    
    # Basic proof information
    proof_info = {
        "theorem_name": proof.get_theorem_name(),
        "theorem_text": proof.theorem.term.text,
        "proof_idx": proof_idx,
        "step_idx": step_idx,
        "step_text": step.step.text,
        "step_number": step.n_step,
        "total_steps_in_proof": len(proof.steps),
        "file_path": dataset_file.file_context.file,
        
        # Context information
        "available_premises": len(dataset_file.get_premises_before(proof)),
        "in_file_premises": len(dataset_file.get_in_file_premises_before(proof)),
        
        # Goals at this step (useful for complexity analysis)
        "num_goals": len(step.goals),
        "goals": [goal.to_string() for goal in step.goals]
    }
    
    return proof_info

def load_and_analyze_dataset(data_loc: Path, sentence_db_loc: Path):
    """Load dataset and analyze all proof steps"""
    sentence_db = SentenceDB.load(sentence_db_loc)
    
    # Load all dataset files - they have .v extensions but contain JSON data
    data_points_loc = data_loc / "data_points"
    all_steps = []
    
    # Changed from *.json to *.v since the files have .v extensions
    for dp_file in data_points_loc.glob("*.v"):
        try:
            dataset_file = DatasetFile.load(dp_file, sentence_db)
            
            for proof_idx, proof in enumerate(dataset_file.proofs):
                for step_idx, step in enumerate(proof.steps):
                    step_info = analyze_proof_step_complexity(dataset_file, proof_idx, step_idx)
                    all_steps.append(step_info)
                    
            print(f"Processed {dp_file.name}: {len(dataset_file.proofs)} proofs, {sum(len(p.steps) for p in dataset_file.proofs)} steps")
            
        except Exception as e:
            print(f"Error processing {dp_file.name}: {e}")
            continue
    
    sentence_db.close()
    print(f"Total analyzed: {len(all_steps)} proof steps from {len(list(data_points_loc.glob('*.v')))} files")
    return all_steps

def calculate_dataset_metrics(all_steps: list[dict]):
    """
    Calculate metrics for the dataset.
    """
    # Calculate the average number of steps per proof
    average_steps_per_proof = sum(step["total_steps_in_proof"] for step in all_steps) / len(all_steps)

    # Calculate the average number of steps per file
    return {
        "total_steps": len(all_steps),
        "average_steps_per_proof": average_steps_per_proof,
    }

def create_curriculum(all_steps: list[dict], data_loc: Path, sentence_db_loc: Path, output_path: Path):
    """
    Create a curriculum from the analyzed proof steps.
    """
    sorted_steps = sorted(all_steps, key=lambda x: x["total_steps_in_proof"])
    curriculum_metadata = {
        "total_steps": len(sorted_steps),
        "shortest_proof": sorted_steps[0]["total_steps_in_proof"],
        "longest_proof": sorted_steps[-1]["total_steps_in_proof"],
        "average_proof_length": sum(s["total_steps_in_proof"] for s in sorted_steps) / len(sorted_steps),
        "curriculum_order": "proof_length_then_step_position"
    }
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "curriculum_metadata.json", "w") as f:
        json.dump(curriculum_metadata, f, indent=2)
    
    with open(output_path / "curriculum_steps.json", "w") as f:
        json.dump(sorted_steps, f, indent=2)
    
    sentence_db = SentenceDB.load(sentence_db_loc)

    premise_conf = SparseConf(
        kind=SparseKind.TFIDF,
        context_format_alias="basic",
        premise_format_alias="basic",
        premise_filter_conf=PremiseFilterConf(
            coq_excludes=[],
            non_coq_excludes=[],
            general_excludes=[],
        ),
        sentence_db_loc=sentence_db_loc,
        cached_premise_loc=None,
    )

    # TODO Check configuration
    formatter = GeneralFormatter.from_conf(
        GeneralFormatterConf(
            premise_client_conf=premise_conf,
            proof_retriever_conf=None,
            num_premises=50,
            num_proofs=None
        )
    )

    example_db = ExampleDB.create(output_path / "curriculum_train.db")
    print("Creating curriculum-ordered training examples...")
    examples_added = 0
    for step in sorted_steps:
        try:
            dp_file = data_loc / "data_points" / step["dp_file"]
            dataset_file = DatasetFile.load(dp_file, sentence_db)

            example = formatter.example_from_step(
                step["step_idx"], 
                step["proof_idx"], 
                dataset_file, 
                training=True
            )
            example_db.insert_example(example.to_json())
            examples_added += 1
            if examples_added % 1000 == 0:
                print(f"Added {examples_added} examples")
        except Exception as e:
            print(f"Error processing {step['theorem_name']}: {e}")
            continue

    sentence_db.close()
    example_db.close()
    print(f"Created {examples_added} curriculum-ordered training examples at: {output_path}")
    return output_path / "curriculum_train.db"

if __name__ == "__main__":
    # Note: The sentence_db_loc should point to the actual database file, not a JSON file
    # The database is typically a .db file, not .json
    # data_loc = Path("raw-data/coq-dataset/")
    # sentence_db_loc = Path("raw-data/coq-dataset/sentences.db")
    # output_path = Path("curriculum_dataset/")
    
    # all_steps = load_and_analyze_dataset(
    #     Path("raw-data/coq-dataset/"), 
    #     Path("raw-data/coq-dataset/ sentences.db")  # Changed from .json to .db
    # ) 
    # curriculum_path = create_curriculum(all_steps, data_loc, sentence_db_loc, output_path)
    # print(f"Curriculum created at: {curriculum_path}")

    exit()