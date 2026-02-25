import json
import os

from tabulate import tabulate


class Logger:
    def __init__(self, name, log_dir=None):
        self.name = name
        self.log_dir = log_dir
        if self.log_dir is None:
            self.log_dir = os.getenv("RESULTS_DIR", os.getcwd())
        self.log_path = os.path.join(self.log_dir, f"{self.name}.jsonl")
        self.staged = {}
        self.reductions = {}

    def stage(self, key, value, reduction=None):
        if key not in self.staged:
            self.staged[key] = []
            self.reductions[key] = reduction
        elif reduction != self.reductions[key]:
            raise ValueError(
                f"Reduction {reduction} does not match previous "
                f"reduction {self.reductions[key]} for key {key}"
            )
        elif self.reductions[key] is None:
            raise ValueError("Cannot log multiple values without reducing")
        self.staged[key].append(value)

    def log(self):
        to_log = {}
        for key, values in self.staged.items():
            reduction = self.reductions[key]
            if reduction is None:
                [value] = values
            elif reduction == "sum":
                value = sum(values)
            elif reduction == "mean":
                value = sum(values) / len(values)
            elif reduction == "min":
                value = min(values)
            elif reduction == "max":
                value = max(values)
            else:
                raise ValueError(f"Unrecognized reduction {reduction}")
            to_log[key] = value
        print(tabulate([[k, v] for k, v in to_log.items()], tablefmt="outline"))
        with open(self.log_path, "a") as log_file:
            json.dump(to_log, log_file)
            log_file.write("\n")
        self.staged = {}
        self.reductions = {}
