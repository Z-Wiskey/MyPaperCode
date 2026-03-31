DATASET_CONFIGS = {
    "NewYork": {
        "aliases": {"ny", "nyc", "newyork", "new_york"},
        "data_dir": "data_NewYork",
        "task_package": "tasks_common",
        "regions_num": 180,
    },
    "Chicago": {
        "aliases": {"chi", "chicago"},
        "data_dir": "data_Chicago",
        "task_package": "tasks_common",
        "regions_num": 77,
    },
    "SanFrancisco": {
        "aliases": {"sf", "sanfrancisco", "san_francisco", "san-francisco"},
        "data_dir": "data_SanFrancisco",
        "task_package": "tasks_common",
        "regions_num": 175,
    },
}

ALL_TASKS = ("crime", "checkin", "servicecall")

TASK_MODULES = {
    "crime": "tasks_crime",
    "checkin": "tasks_checkin",
    "servicecall": "tasks_servicecall",
}

TASK_ALIASES = {
    "crime": "crime",
    "check": "checkin",
    "checkin": "checkin",
    "check_in": "checkin",
    "check-in": "checkin",
    "chk": "checkin",
    "servicecall": "servicecall",
    "service_call": "servicecall",
    "service-call": "servicecall",
    "service": "servicecall",
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
