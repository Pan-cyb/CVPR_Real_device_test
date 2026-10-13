"""Simulated environments beyond the DeepMind Control Suite.

``load_environment`` resolves a domain/task pair the way ``suite.load`` does:
domains registered here come first, everything else goes to dm_control.
"""

from __future__ import annotations

from dm_control import suite

from . import rm65b_cartpole, rm65b_pendulum


def _load_rm65b(task: str, task_kwargs=None):
    if task == "pendulum":
        return rm65b_pendulum.load(task, task_kwargs=task_kwargs)
    if task == "cartpole":
        return rm65b_cartpole.load(task, task_kwargs=task_kwargs)
    raise ValueError(f"Unknown rm65b task: {task}")


DOMAINS = {"rm65b": _load_rm65b}


def load_environment(domain: str, task: str, task_kwargs=None):
    if domain in DOMAINS:
        return DOMAINS[domain](task, task_kwargs=task_kwargs)
    return suite.load(domain, task, task_kwargs=task_kwargs)


__all__ = ["DOMAINS", "load_environment"]
