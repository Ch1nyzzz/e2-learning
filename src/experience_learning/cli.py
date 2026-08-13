from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from experience_learning.config import EnvironmentConfig, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experience-learning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run active ALFWorld experience learning")
    train.add_argument("--config", required=True)
    train.add_argument("--set", action="append", default=[], dest="overrides")
    collect_rwml = subparsers.add_parser(
        "collect-rwml-data", help="collect fixed on-policy ALFWorld transitions for RWML baselines"
    )
    collect_rwml.add_argument("--config", required=True)
    collect_rwml.add_argument("--set", action="append", default=[], dest="overrides")
    collect_rwml.add_argument("--tasks", type=int, default=2048)
    collect_rwml.add_argument("--rollouts-per-task", type=int, default=3)
    collect_rwml.add_argument("--parallel-environments", type=int, default=8)
    collect_rwml.add_argument("--max-steps", type=int, default=30)
    collect_rwml.add_argument("--max-action-tokens", type=int, default=512)
    collect_rwml.add_argument("--output", required=True)
    split_rwml = subparsers.add_parser(
        "split-rwml-data", help="make a deterministic shared train/validation transition split"
    )
    split_rwml.add_argument("--config", required=True)
    split_rwml.add_argument("--set", action="append", default=[], dest="overrides")
    split_rwml.add_argument("--input", required=True)
    split_rwml.add_argument("--train-output", required=True)
    split_rwml.add_argument("--validation-output", required=True)
    split_rwml.add_argument("--validation-fraction", type=float, default=0.1)
    subset_rwml = subparsers.add_parser(
        "subset-rwml-data", help="make a deterministic transition-count-matched subset"
    )
    subset_rwml.add_argument("--config", required=True)
    subset_rwml.add_argument("--set", action="append", default=[], dest="overrides")
    subset_rwml.add_argument("--input", required=True)
    subset_rwml.add_argument("--output", required=True)
    subset_rwml.add_argument("--records", type=int, required=True)
    wm_sft = subparsers.add_parser(
        "train-wm-sft", help="full-parameter offline WM SFT on a fixed transition corpus"
    )
    wm_sft.add_argument("--config", required=True)
    wm_sft.add_argument("--set", action="append", default=[], dest="overrides")
    wm_sft.add_argument("--train-data", required=True)
    wm_sft.add_argument("--validation-data")
    wm_sft.add_argument("--epochs", type=int, default=2)
    wm_sft.add_argument("--effective-batch-size", type=int, default=32)
    validate = subparsers.add_parser("validate-config", help="resolve and validate configuration")
    validate.add_argument("--config", required=True)
    validate.add_argument("--set", action="append", default=[], dest="overrides")
    collect = subparsers.add_parser(
        "collect-probes", help="collect a fixed random held-out transition set"
    )
    collect.add_argument("--config", required=True)
    collect.add_argument("--set", action="append", default=[], dest="overrides")
    collect.add_argument("--split", required=True)
    collect.add_argument("--episodes", type=int, required=True)
    collect.add_argument("--output", required=True)
    evaluate = subparsers.add_parser(
        "evaluate-probes", help="evaluate semantic accuracy and target-token NLL"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--set", action="append", default=[], dest="overrides")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--probes", required=True)
    evaluate.add_argument("--output", required=True)
    success = subparsers.add_parser(
        "evaluate-sr", help="evaluate deterministic ALFWorld task success rate"
    )
    success.add_argument("--config", required=True)
    success.add_argument("--set", action="append", default=[], dest="overrides")
    success.add_argument("--checkpoint")
    success.add_argument(
        "--split",
        required=True,
        choices=["eval_in_distribution", "eval_out_of_distribution"],
    )
    success.add_argument("--episodes", type=int, default=0)
    success.add_argument("--max-steps", type=int, default=30)
    success.add_argument("--report-step", type=int, default=20)
    success.add_argument("--max-action-tokens", type=int, default=256)
    success.add_argument("--seed", type=int, default=42)
    success.add_argument("--output", required=True)
    return parser


def _write_resolved_config(config: object, output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    if args.command == "validate-config":
        print(json.dumps(asdict(config), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "collect-probes":
        from experience_learning.evaluation import collect_random_probes

        result = collect_random_probes(
            config,
            split=args.split,
            episodes=args.episodes,
            output_path=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "split-rwml-data":
        from experience_learning.offline import split_rwml_transitions

        result = split_rwml_transitions(
            input_path=args.input,
            train_path=args.train_output,
            validation_path=args.validation_output,
            validation_fraction=args.validation_fraction,
            seed=config.experiment.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "subset-rwml-data":
        from experience_learning.offline import make_deterministic_subset

        result = make_deterministic_subset(
            input_path=args.input,
            output_path=args.output,
            records=args.records,
            seed=config.experiment.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    from experience_learning.distributed import broadcast_object
    from experience_learning.environment import ALFWorldTextEnvironment
    from experience_learning.experiment import OnlineExperienceExperiment
    from experience_learning.judge import build_judge
    from experience_learning.model import TransformersWorldModel

    if args.command in {"evaluate-probes", "evaluate-sr"} and args.checkpoint:
        config.training.resume_from = args.checkpoint
    training_command = args.command in {"train", "train-wm-sft"}
    model = TransformersWorldModel(config, training=training_command)
    is_main = model.accelerator.is_main_process
    if args.command == "collect-rwml-data":
        from experience_learning.offline import collect_rwml_transitions

        def environment_factory(slot: int, stride: int) -> ALFWorldTextEnvironment:
            env_config = EnvironmentConfig(**vars(config.environment))
            env_config.split = "train"
            env_config.max_steps_per_episode = args.max_steps
            return ALFWorldTextEnvironment(
                env_config,
                config.experiment.seed + slot,
                game_offset=slot,
                game_stride=stride,
            )

        result = collect_rwml_transitions(
            model=model,
            environment_factory=environment_factory,
            is_main_process=is_main,
            tasks=args.tasks,
            rollouts_per_task=args.rollouts_per_task,
            parallelism=args.parallel_environments,
            max_steps=args.max_steps,
            max_action_tokens=args.max_action_tokens,
            excluded_actions=config.environment.excluded_actions,
            seed=config.experiment.seed,
            output_path=args.output,
        )
        if is_main:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "train-wm-sft":
        from experience_learning.offline import train_offline_wm_sft

        _write_resolved_config(config, config.experiment.output_dir)
        result = train_offline_wm_sft(
            model=model,
            is_main_process=is_main,
            train_path=args.train_data,
            validation_path=args.validation_data,
            output_dir=config.experiment.output_dir,
            epochs=args.epochs,
            effective_batch_size=args.effective_batch_size,
            seed=config.experiment.seed,
            evaluation_batch_size=config.evaluation.batch_size,
        )
        if is_main:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate-sr":
        from experience_learning.success_evaluation import evaluate_success_rate

        official_counts = {
            "eval_in_distribution": 140,
            "eval_out_of_distribution": 134,
        }
        episodes = args.episodes or official_counts[args.split]
        parallelism = min(config.experiment.parallel_environments, episodes)
        environments = []
        startup = None
        if is_main:
            try:
                env_config = EnvironmentConfig(**vars(config.environment))
                env_config.split = args.split
                env_config.max_steps_per_episode = args.max_steps
                for index in range(parallelism):
                    environments.append(
                        ALFWorldTextEnvironment(
                            env_config,
                            args.seed + index,
                            game_offset=index,
                            game_stride=parallelism,
                        )
                    )
                startup = {"phase": "OK"}
            except Exception as exc:
                for environment in environments:
                    environment.close()
                startup = {
                    "phase": "STOP",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        startup = broadcast_object(startup)
        if startup["phase"] == "STOP":
            raise RuntimeError(
                f"SR evaluation startup failed: {startup['error_type']}: {startup['error']}"
            )
        result = evaluate_success_rate(
            model=model,
            environments=environments,
            is_main_process=is_main,
            split=args.split,
            episodes=episodes,
            parallelism=parallelism,
            max_steps=args.max_steps,
            report_step=args.report_step,
            max_action_tokens=args.max_action_tokens,
            seed=args.seed,
            output_path=args.output,
        )
        if is_main:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate-probes":
        from experience_learning.evaluation import evaluate_probes

        judge = None
        startup = None
        if is_main:
            try:
                judge = build_judge(config.judge)
                startup = {"phase": "OK"}
            except Exception as exc:
                startup = {
                    "phase": "STOP",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        startup = broadcast_object(startup)
        if startup["phase"] == "STOP":
            raise RuntimeError(
                f"evaluation startup failed: {startup['error_type']}: {startup['error']}"
            )
        result = evaluate_probes(
            config,
            model=model,
            judge=judge,
            is_main_process=is_main,
            probes_path=args.probes,
            output_path=args.output,
        )
        if is_main:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    environments = []
    judge = None
    startup = None
    if is_main:
        try:
            parallelism = config.experiment.parallel_environments
            for index in range(parallelism):
                environments.append(
                    ALFWorldTextEnvironment(
                        config.environment,
                        config.experiment.seed + index,
                        game_offset=index,
                        game_stride=parallelism,
                    )
                )
            judge = build_judge(config.judge)
            _write_resolved_config(config, config.experiment.output_dir)
            startup = {"phase": "OK"}
        except Exception as exc:
            for environment in environments:
                environment.close()
            startup = {
                "phase": "STOP",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    startup = broadcast_object(startup)
    if startup["phase"] == "STOP":
        raise RuntimeError(f"training startup failed: {startup['error_type']}: {startup['error']}")
    result = OnlineExperienceExperiment(
        config=config,
        model=model,
        is_main_process=is_main,
        environment=environments,
        judge=judge,
    ).run()
    if is_main:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
