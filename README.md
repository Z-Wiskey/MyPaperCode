# MyPaperCode

This project is now organized in a HAFusion-style runtime layout while keeping the HRE model code unchanged.

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
python main.py --city NewYork --task check
python main.py --city NewYork --task clustering
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
