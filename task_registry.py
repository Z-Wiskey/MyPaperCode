from functools import lru_cache
from importlib import import_module

from project_config import ALL_TASKS, TASK_MODULES


@lru_cache(maxsize=None)
def load_task_module(task_package, task_name):
    if task_name not in TASK_MODULES:
        supported = ", ".join(ALL_TASKS)
        raise ValueError(f"Unsupported task '{task_name}'. Supported values: {supported}.")
    return import_module(f"{task_package}.{TASK_MODULES[task_name]}")


def run_task(task_package, task_name, embs, data_path, display=True):
    task_module = load_task_module(task_package, task_name)
    return task_module.do_tasks(embs, data_path=data_path, display=display)


def expand_tasks(task_name):
    return ALL_TASKS if task_name == "all" else (task_name,)
