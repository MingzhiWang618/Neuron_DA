"""Train the fully supervised CREMI-A contrastive oracle."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from connectomics.config import load_config, resolve_default_profiles, validate_config
from connectomics.config.hardware import resolve_runtime_resource_sentinels
from connectomics.config.pipeline import resolve_data_paths
from connectomics.runtime.preflight import validate_runtime_coherence
from connectomics.training.lightning import create_datamodule, create_trainer, setup_run_directory

from module import ContrastiveOracleModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--contrastive-config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    cfg = resolve_default_profiles(load_config(args.config), mode="train")
    cfg = resolve_data_paths(cfg)
    cfg = resolve_runtime_resource_sentinels(cfg, print_results=True)
    validate_config(cfg)
    validate_runtime_coherence(cfg)
    contrastive_cfg = OmegaConf.load(args.contrastive_config)

    output_base = str(Path(cfg.save_path))
    run_dir = setup_run_directory("train", cfg, output_base, args.checkpoint)
    OmegaConf.save(contrastive_cfg, run_dir / "contrastive.yaml")
    model = ContrastiveOracleModule(cfg, contrastive_cfg)
    data = create_datamodule(cfg, mode="train", fast_dev_run=args.fast_dev_run)
    trainer = create_trainer(
        cfg,
        run_dir=run_dir,
        fast_dev_run=args.fast_dev_run,
        ckpt_path=args.checkpoint,
        mode="train",
    )
    trainer.fit(model, datamodule=data, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
