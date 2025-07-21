#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import json
import sqlite3
from typing import List, Dict, Any, Tuple, Optional
import re
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from tactic_gen.lm_example import GeneralFormatter, GeneralFormatterConf

from data_management.dataset_file import DatasetFile
from data_management.sentence_db import SentenceDB
from data_management.jsonl_utils import ExampleDB
from premise_selection.premise_conf import SparseConf, PremiseFilterConf, SparseKind

DATA_POINTS_PATH = "raw-data/coq-dataset/data_points/"

def sort_data_by_proof_length(
        db_path: Path, 
        proof_length: bool=True, 
        proof_state_length: bool=True,
    ) -> List[dict]:
    # Data in db is sorted first by proof length and then by proof_state_length
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM example"
        if proof_length:
            query += " ORDER BY proof_length DESC"
        if proof_state_length:
            query += ", proof_state_length DESC"
        cursor.execute(query)
        examples = cursor.fetchall()
    return examples

#TODO: Currently only proof length is retrieved from the raw data.
# Possibly retrieve other information from the raw data
def get_data_from_raw_data(file_name: str, proof_idx: int=None) -> Optional[dict]:
    data = {}
    file_data_point_path = DATA_POINTS_PATH + file_name

    with open(file_data_point_path, "r") as f:
        content = json.load(f)

    if proof_idx is not None:
        proofs = content["proofs"]
        current_proof = None
        for proof in proofs:
            if proof["theorem"]["id"] == proof_idx:
                current_proof = proof
                break
        if current_proof is not None:
            print("proof not found with id", proof_idx)
            return None
        proof_length = len(current_proof["steps"])
        return proof_length
    
def get_steps_from_proof(db_path: Path, proof_idx: int) -> List[dict]:
    steps = []
    # Database already indexed by proof_idx
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM example WHERE proof_idx = ?", (proof_idx,))
        steps = cursor.fetchall()
    return steps

def update_db_with_proof_state_length(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE example ADD COLUMN proof_state_length INTEGER")
        conn.commit()
         # Query the maximum proof_idx
        cursor.execute("SELECT MAX(proof_idx) FROM example;")
        max_proof_idx = cursor.fetchone()[0]    

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for proof_idx in range(max_proof_idx):
            steps = get_steps_from_proof(db_path, proof_idx)
            for step in steps:
                step_idx = step["step_idx"]
                proof_state = step["proof_state"]
                cursor.execute("UPDATE example SET proof_state_length = ? WHERE proof_idx = ? AND step_idx = ?", (len(proof_state), proof_idx, step_idx))
            conn.commit()

def update_db_with_proof_length(db_path: Path):
    #Add new column "proof_length" to the database if it doesn't exist
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE example ADD COLUMN proof_length INTEGER")
        conn.commit()
         # Query the maximum proof_idx
        cursor.execute("SELECT MAX(proof_idx) FROM example;")
        max_proof_idx = cursor.fetchone()[0]
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for proof_idx in range(max_proof_idx):
            proof_length = get_data_from_raw_data(db_path, proof_idx)
            if proof_length is not None:
                cursor.execute("UPDATE example SET proof_length = ? WHERE proof_idx = ?", (proof_length, proof_idx))
            conn.commit()



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

    # Test the function
    test_proof_length_functions()

    exit()