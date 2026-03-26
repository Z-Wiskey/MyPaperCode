# MyPaperCode

This project is organized in a HAFusion-style runtime layout while keeping the HRE model code unchanged.
The New York experiment now uses HAFusion-style structural inputs: POI similarity, mobility adjacency, land-use similarity, and the original visual embedding.

## Structure

- `data_NewYork/`: New York dataset files used by the current project
- `tasks_NewYork/`: downstream task scripts for the New York dataset
- `project_config.py`: dataset and task registry
- `parse_args.py`: unified CLI parser
- `HRE_Module_Train.py`: training entry with dataset/task dispatch
- `main.py`: simple launcher

## Quick Start

Train for a single downstream task:

```bash
python main.py --city NewYork --task crime
python main.py --city NewYork --task checkin
python main.py --city NewYork --task servicecall
```

Run all downstream tasks after training, while selecting the best checkpoint with a specific task:

```bash
python main.py --city NewYork --task all --selection_task crime
```

Useful runtime parameters:

```bash
python main.py --city NY --task crime --epochs 500 --learning_rate 0.0002 --dropout 0.2
```

Outputs are saved under `outputs/<city>/<task>/` by default.
