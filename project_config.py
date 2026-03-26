DATASET_CONFIGS = {
    "NewYork": {
        "aliases": {"ny", "nyc", "newyork", "new_york"},
        "data_dir": "data_NewYork",
        "task_package": "tasks_NewYork",
        "regions_num": 180,
    },
}

ALL_TASKS = ("crime", "check", "clustering")

TASK_MODULES = {
    "crime": "tasks_crime",
    "check": "tasks_check",
    "clustering": "tasks_clustering",
}

TASK_ALIASES = {
    "crime": "crime",
    "check": "check",
    "checkin": "check",
    "check_in": "check",
    "check-in": "check",
    "chk": "check",
    "clustering": "clustering",
    "cluster": "clustering",
    "all": "all",
}

DATASET_ALIASES = {}
for canonical_name, config in DATASET_CONFIGS.items():
    DATASET_ALIASES[canonical_name.lower()] = canonical_name
    for alias in config["aliases"]:
        DATASET_ALIASES[alias.lower()] = canonical_name


def canonicalize_city(city_name):
    normalized = city_name.strip().lower()
    if normalized not in DATASET_ALIASES:
        supported = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unsupported city/dataset '{city_name}'. Supported values: {supported}.")
    return DATASET_ALIASES[normalized]


def canonicalize_task(task_name):
    normalized = task_name.strip().lower()
    if normalized not in TASK_ALIASES:
        supported = ", ".join(list(ALL_TASKS) + ["all"])
        raise ValueError(f"Unsupported task '{task_name}'. Supported values: {supported}.")
    return TASK_ALIASES[normalized]


def get_dataset_config(city_name):
    canonical_name = canonicalize_city(city_name)
    return DATASET_CONFIGS[canonical_name].copy()
