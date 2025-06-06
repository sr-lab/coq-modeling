import os
import logging
import uuid

from sentence_transformers import SentenceTransformer, util
from coqpyt.lsp.structs import (
    VersionedTextDocumentIdentifier,
    TextDocumentContentChangeEvent,
    Position,
    TextDocumentIdentifier
)

DESIDER_REASONING_LENGTH = 1024
MAX_REASONING_LENGTH = 4096
DESIDER_TACTIC_LENGTH = 128

model = SentenceTransformer('nomic-ai/CodeRankEmbed', trust_remote_code=True).to('cpu')

def calculate_reasoning_format_reward(prompts, completions, answer, **kwargs):
    rewards = []
    for completion in completions:
        completion = completion.strip()
        if '<think>' in completion and '</think>' in completion:
            think_start = completion.find('<think>')
            think_end = completion.find('</think>')
            if think_start > think_end:
                rewards.append(-1)
            else:
                rewards.append(0)
        else:
            rewards.append(-1)
   
    return rewards

# def calculate_reasoning_length_reward(prompts, completions, answer, **kwargs):
#     rewards = []
#     for completion in completions:
#         completion = completion.strip()
#         reasoning = completion.split('<think>')[-1].split('</think>')[0].strip()
#         reasoning_length = len(reasoning)
#         if reasoning_length < DESIDER_REASONING_LENGTH:
#             rewards.append(0)
#         else:
#             rewards.append(-1)
#     print("Rewards reasoning length", rewards, len(rewards))
#     return rewards

# def calculate_tactic_format_reward(prompts, completions, answer, **kwargs):
#     rewards = []
#     for completion in completions:
#         completion = completion.split("</think>")[-1].strip()
#         rewards.append(0 if completion.endswith('.') and completion.count('.') == 1 else -1)
#     print("Rewards tactic format", rewards, len(rewards))
#     return rewards

# def calculate_tactic_length_reward(prompts, completions, answer, **kwargs):
#     rewards = []
#     for completion in completions:
#         completion = completion.split("</think>")[-1].strip()
#         rewards.append(0 if len(completion) < DESIDER_TACTIC_LENGTH else -1)
#     print("Rewards tactic length", rewards, len(rewards))
#     return rewards



def goals_exist(goals):
    return (
        (goals is not None and goals.goals is not None and goals.goals.goals is not None)
    )


def calculate_unchanged_reward(previous_goals, final_goals):
    if repr(previous_goals) == repr(final_goals):
        return -1
    return 0

def reward_goals(ground_truth_goals, final_goals):
    """
    Compare the initial goals with the final goals.
    """
    if (not goals_exist(final_goals)):
        return 2
    elif ((goals_exist(ground_truth_goals) and len(ground_truth_goals.goals.goals) == 0) 
          and len(final_goals.goals.goals) == 0
          ):
        return 2
    elif ((goals_exist(ground_truth_goals) and len(ground_truth_goals.goals.goals) == 0) 
          and len(final_goals.goals.goals) != 0
          ):
        return 0
    elif len(ground_truth_goals.goals.goals) > len(final_goals.goals.goals):
        return 2
    elif len(ground_truth_goals.goals.goals) < len(final_goals.goals.goals):
        return 0
    else:
        goals_reward_ty = 0
        goals_reward_hyps = 0
        for i in range(len(ground_truth_goals.goals.goals)):
            embedding1_ty = model.encode(ground_truth_goals.goals.goals[i].ty, convert_to_tensor=True)
            embedding2_ty = model.encode(final_goals.goals.goals[i].ty, convert_to_tensor=True)
            similarity_ty = util.cos_sim(embedding1_ty, embedding2_ty).item()

            hyps = list(map(lambda hyp: repr(hyp), ground_truth_goals.goals.goals[i].hyps))
            embedding1_hyps = model.encode("".join(hyps), convert_to_tensor=True)
            hyps = list(map(lambda hyp: repr(hyp), final_goals.goals.goals[i].hyps))
            embedding2_hyps = model.encode("".join(hyps), convert_to_tensor=True)
            similarity_hyps = util.cos_sim(embedding1_hyps, embedding2_hyps).item()

            goals_reward_ty += similarity_ty
            goals_reward_hyps += similarity_hyps

        return (
            goals_reward_ty / len(ground_truth_goals.goals.goals) + 
            goals_reward_hyps / len(ground_truth_goals.goals.goals)
        )


def get_last_point(s: str) -> tuple[int, int]:
    """
    Returns the (line, column) of the last character in the string.
    Lines and columns are 0-based.
    If the string is empty, returns (0, 0).
    """
    if not s:
        return (0, 0)
    lines = s.splitlines(keepends=True)
    if not lines:
        return (0, 0)
    last_line_idx = len(lines) - 1
    last_line = lines[-1]
    # If the string ends with a newline, the last character is at column 0 of the next line
    if last_line.endswith('\n') or last_line.endswith('\r'):
        return (last_line_idx + 1, 0)
    else:
        return (last_line_idx, len(last_line))
    
def get_file_info(conf, file_name, proof_script):
    with open(os.path.join(conf["repos_path"], file_name), "r") as f:
        file_contents = f.read()
        theorem_split = file_contents.split(proof_script.split(":")[0])
        if len(theorem_split) == 0:
            logging.error("Proof script not found in file %s", file_name)
            exit(-1)
        prefix = theorem_split[0] + "\n" + proof_script
    
    original_file_path = os.path.join(conf["repos_path"], file_name)
    dir_path = os.path.dirname(original_file_path)
    file_basename = os.path.basename(file_name)
    random_id = str(uuid.uuid4())[:8]
    temp_file_name = os.path.join(dir_path, f"temp_{random_id}_{file_basename}")
    
    return temp_file_name, prefix

def get_proof_goals(coq_file, line, column, uri):
    goals = coq_file.coq_lsp_client.proof_goals(
            TextDocumentIdentifier(uri),
            Position(line, column+1)
        )
    return goals
