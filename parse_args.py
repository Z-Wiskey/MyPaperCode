import argparse
import os

from project_config import canonicalize_city, canonicalize_task, get_dataset_config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train HRE with selectable dataset and downstream task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------Runtime--------------------------- #
    parser.add_argument(
        "--city",
        "--dataset",
        dest="city",
        default="NewYork",
        help="Dataset/city name. Aliases such as NY, Chi, and SF are also supported.",
    )
    parser.add_argument(
        "--task",
        default="crime",
        help="Downstream task name: crime, checkin, servicecall, or all.",
    )
    parser.add_argument(
        "--selection_task",
        default=None,
        help="Task used to pick the best checkpoint when --task is all.",
    )
    parser.add_argument("--data_path", default=None, help="Optional explicit dataset directory.")
    parser.add_argument("--output_root", default="./outputs", help="Root directory for run outputs.")
    parser.add_argument("--save_folder", default=None, help="Optional explicit output directory.")

    # ---------------------------File--------------------------- #
    parser.add_argument("--vis_embedding", default="vis_embedding.npy")
    parser.add_argument("--mobility_adj", default="mob-adj.npy")
    parser.add_argument("--poi_similarity", default="poi_simi.npy")
    parser.add_argument("--landuse_similarity", default="landUse_simi.npy")
    parser.add_argument("--crime_counts", default="crime_counts.npy")
    parser.add_argument("--checkin_counts", default="check_counts.npy")
    parser.add_argument("--servicecall_counts", default="serviceCall_counts.npy")

    # ---------------------------Model--------------------------- #
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--embedding_size", type=int, default=144)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--feature_learning_rate", type=float, default=0.001)
    parser.add_argument("--aux_learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--regions_num", type=int, default=None)
    parser.add_argument("--importance_k", type=int, default=10)
    parser.add_argument("--volume_loss_weight", type=float, default=0.5)
    parser.add_argument("--cross_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--selection_strategy",
        choices=("single", "pareto", "baseline_pareto"),
        default="single",
        help="Checkpoint selection logic for multi-task runs.",
    )
    parser.add_argument(
        "--reference_emb_path",
        default=None,
        help="Optional embedding file used as a reference checkpoint for baseline-relative selection.",
    )
    parser.add_argument(
        "--fusion_mode",
        choices=("attention", "dafusion"),
        default="attention",
        help="Structural feature fusion module.",
    )
    parser.add_argument("--dafusion_heads", type=int, default=4)
    parser.add_argument("--dafusion_inter_blocks", type=int, default=3)
    parser.add_argument("--dafusion_region_blocks", type=int, default=3)
    parser.add_argument("--dafusion_inter_slot_dim", type=int, default=72)
    parser.add_argument("--dafusion_view_hidden_dim", type=int, default=64)

    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.city = canonicalize_city(args.city)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        args.task = canonicalize_task(args.task)
    except ValueError as exc:
        parser.error(str(exc))

    selection_task = args.selection_task or ("crime" if args.task == "all" else args.task)
    try:
        args.selection_task = canonicalize_task(selection_task)
    except ValueError as exc:
        parser.error(str(exc))

    if args.selection_task == "all":
        parser.error("--selection_task cannot be set to all.")

    dataset_config = get_dataset_config(args.city)
    if args.data_path is None:
        args.data_path = os.path.join(".", dataset_config["data_dir"])
    args.task_package = dataset_config["task_package"]
    if args.regions_num is None:
        args.regions_num = dataset_config["regions_num"]

    if args.save_folder is None:
        args.save_folder = os.path.join(args.output_root, args.city, args.task)

    args.best_emb_path = os.path.join(args.save_folder, "best_emb.npy")
    args.best_metrics_path = os.path.join(args.save_folder, "best_metrics.json")
    return args


args = parse_args()
