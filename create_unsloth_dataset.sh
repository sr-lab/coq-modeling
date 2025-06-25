#!/bin/bash

source venv/bin/activate
export OPENAI_API_KEY=""
eval $(opam env)


python src/tactic_gen/create_unsloth_dataset.py confs/train/sft_conf.yaml --split train --output-path unsloth_sft_dataset
