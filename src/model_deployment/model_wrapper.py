from __future__ import annotations
from typing import Any, Optional
from pathlib import Path
import yaml

from enum import Enum

from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
)
import torch

from util.train_utils import get_required_arg

from tactic_gen.lm_example import (
    LmExample,
)
from tactic_gen.train_decoder import (
    get_tokenizer,
    get_model,
)
from tactic_gen.tactic_data import (
    ExampleCollator,
    ProofPremiseCollator,
    NoScriptCollator,
    ReasoningCollator,
    example_collator_from_conf,
    example_collator_conf_from_yaml,
    NEWLINE_RESPONSE_TEMPLATE,
)
from model_deployment.model_result import ModelResult, filter_recs

from vllm import SamplingParams
inference = True
if inference:
    print(" \n\n============= Inference mode =============\n\n ")

class TokenMask(Enum):
    STATE = 0
    SCRIPT = 1
    PROOF = 2
    PREMISE = 3

    @classmethod
    def from_str(cls, s: str) -> TokenMask:
        match s:
            case "state":
                return cls.STATE
            case "script":
                return cls.SCRIPT
            case "proof":
                return cls.PROOF
            case "premise":
                return cls.PREMISE
            case _:
                raise ValueError(f"Invalid token mask: {s}")


def find_id_start_idx(t: torch.Tensor, s: torch.Tensor) -> Optional[int]:
    for i in range(t.shape[0] - s.shape[0] + 1):
        if torch.all(t[i : i + s.shape[0]] == s):
            return i
    return None


def get_enclosing_seps(
    collator: ExampleCollator, token_mask: TokenMask
) -> tuple[str, str]:
    match collator:
        case ProofPremiseCollator():
            match token_mask:
                case TokenMask.STATE:
                    return (collator.STATE_SEP, collator.SCRIPT_SEP)
                case TokenMask.SCRIPT:
                    return (collator.SCRIPT_SEP, NEWLINE_RESPONSE_TEMPLATE)
                case TokenMask.PROOF:
                    return (collator.PROOF_SEP, collator.STATE_SEP)
                case TokenMask.PREMISE:
                    return (collator.PREMISE_SEP, collator.PROOF_SEP)

        case NoScriptCollator():
            match token_mask:
                case TokenMask.STATE:
                    return (collator.STATE_SEP, NEWLINE_RESPONSE_TEMPLATE)
                case TokenMask.SCRIPT:
                    raise ValueError(
                        "NoScriptCollator does not support SCRIPT token masking."
                    )
                case TokenMask.PROOF:
                    return (collator.PROOF_SEP, collator.STATE_SEP)
                case TokenMask.PREMISE:
                    return (collator.PREMISE_SEP, collator.PROOF_SEP)

        case _:
            raise ValueError(f"Token masking not supported for {collator}.")


def transform_attention_mask(
    collator: ExampleCollator,
    tokenizer: PreTrainedTokenizer,
    token_mask: Optional[TokenMask],
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    if token_mask is None:
        return attn_mask
    start_str, end_str = get_enclosing_seps(collator, token_mask)
    start_ids = tokenizer.encode(start_str, add_special_tokens=False)
    end_ids = tokenizer.encode(end_str, add_special_tokens=False)

    changed_mask = attn_mask.clone()
    for i, id_row in enumerate(input_ids):
        start_idx = find_id_start_idx(id_row, torch.tensor(start_ids))
        end_idx = find_id_start_idx(id_row, torch.tensor(end_ids))
        assert start_idx is not None
        assert end_idx is not None
        changed_mask[i, start_idx:end_idx] = 0
    return changed_mask


class DecoderLocalWrapper:
    ALIAS = "decoder-local"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        collator: ExampleCollator,
        hard_seq_len: int,
        max_new_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.hard_seq_len = hard_seq_len
        self.max_new_tokens = max_new_tokens
        
    def get_recs(
        self,
        example: LmExample,
        n: int,
        current_proof: str,
        beam: bool,
        token_mask_str,
    ) -> ModelResult:
        token_mask = None
        if token_mask_str is not None:
            token_mask = TokenMask.from_str(token_mask_str)
        collated_input = self.collator.collate_input(self.tokenizer, example)
        
        outputs = self.model.generate(collated_input, SamplingParams(max_tokens=self.max_new_tokens))
        tactics = []
        for output in outputs:
            generated_text = output.outputs[0].text
            tactics.append(generated_text)

        return ModelResult(tactics, [], [])

    @classmethod
    def get_training_conf(cls, checkpoint_loc: Path) -> Any:
        training_conf_loc = checkpoint_loc.parent / "training_conf.yaml"
        with training_conf_loc.open("r") as f:
            training_conf = yaml.safe_load(f)
        return training_conf

    @classmethod
    def from_checkpoint(cls, checkpoint_loc: Path, conf: dict[str, Any]) -> DecoderLocalWrapper:
        training_conf = cls.get_training_conf(checkpoint_loc)
        hard_seq_length = get_required_arg("hard_seq_len", training_conf)
        example_collator_conf = example_collator_conf_from_yaml(
            training_conf["example_collator"]
        )
        example_collator = example_collator_from_conf(example_collator_conf)

        model, tokenizer = get_model(str(checkpoint_loc.resolve()), conf)
        
        if tokenizer is None:
            tokenizer = get_tokenizer(
                get_required_arg("model_name", training_conf), add_eos=False
            )

        if "max_new_tokens" in conf:
            max_new_tokens = conf["max_new_tokens"]
        else:
            max_new_tokens = 128
        return cls(
            model, 
            tokenizer, 
            example_collator, 
            hard_seq_length, 
            max_new_tokens
        )

    @classmethod
    def from_conf(cls, json_data: Any) -> DecoderLocalWrapper:
        name = json_data["checkpoint_loc"]
        return cls.from_checkpoint(Path(name), json_data)


class StubWrapper:
    def get_recs(
        self,
        example: LmExample,
        n: int,
        current_proof: str,
        beam: bool,
        token_mask: Optional[str],
    ) -> ModelResult:
        return ModelResult([], [], [])


ModelWrapper = DecoderLocalWrapper | StubWrapper


class WrapperNotFoundError(Exception):
    pass


def wrapper_from_conf(conf: Any) -> ModelWrapper:
    attempted_alias = conf["alias"]
    match attempted_alias:
        case DecoderLocalWrapper.ALIAS:
            return DecoderLocalWrapper.from_conf(conf)
        case _:
            raise WrapperNotFoundError(
                f"Could not find model wrapper: {attempted_alias}"
            )
