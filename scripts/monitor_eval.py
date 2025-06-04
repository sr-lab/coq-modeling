import json
from pathlib import Path
import argparse
import os

from coqstoq.check import Result


def monitor_results(eval_loc: Path, delete: bool):
    num_success = 0
    num_failure = 0
    num_incomplete = 0
    attemps = []

    for p in eval_loc.glob("**/*.json"):
        with p.open() as fin:
            try:
                data = json.load(fin)
                attemps.append(data["n_attempts"])
                r = Result.from_json(data)
                if r.proof is not None:
                    num_success += 1
                elif r.time is None:
                    if delete:
                        os.remove(p)
                        print("deleted:", p)
                    else:
                        print("incomplete:", p)
                    num_incomplete += 1
                else:
                    num_failure += 1
            except json.JSONDecodeError:
                continue

    print(f"Success: {num_success}")
    print(f"Failure: {num_failure}")
    print(f"Incomplete: {num_incomplete}")
    print(f"Total: {num_success + num_failure + num_incomplete}")
    print(
        f"Success rate (w/) incomplete: {num_success / (num_success + num_failure + num_incomplete)}"
    )
    print(f"Success rate (w/o) incomplete: {num_success / (num_success + num_failure)}")

    print(f"Average attempts: {sum(attemps) / len(attemps)}")
    print(f"Median attempts: {sorted(attemps)[len(attemps) // 2]}")
    print(f"Max attempts: {max(attemps)}")
    print(f"Min attempts: {min(attemps)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_loc")
    parser.add_argument("--delete", default=False, action="store_true")
    args = parser.parse_args()
    monitor_results(Path(args.eval_loc), args.delete)
