"""
Hydra-based configuration utilities.

This module provides both decorator-based and programmatic config loading,
maintaining compatibility with the original Config class interface.

Usage:
    # Programmatic loading (for notebooks, tests, etc.)
    from config_utils import load_config
    cfg = load_config(overrides=["experiment=continue_att_backbone", "NUM_GPUS=4"])
    
    # Access config values (same as before)
    print(cfg.DATA.NUM_INPUT_FRAMES)
    print(cfg.OPTIMIZER.BASE_LR)
"""

import os
from pathlib import Path
from typing import Optional, List, Any

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import tadaconv.utils.checkpoint as ckp


def load_config(
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    create_output_dir: bool = True,
) -> DictConfig:
    """
    Load Hydra config programmatically (without decorator).
    
    This function can be called multiple times safely - it clears
    any existing Hydra instance before initializing.
    
    Args:
        config_name: Name of the main config file (without .yaml)
        overrides: List of overrides like:
            - ["NUM_GPUS=4"]  # Simple override
            - ["experiment=continue_att_backbone"]  # Select experiment
            - ["training=backbone_soccer_att"]  # Select training config
            - ["+new_key=value"]  # Add new key
        config_path: Absolute path to config directory. Defaults to ./configs
        create_output_dir: Whether to create OUTPUT_DIR if it doesn't exist
    
    Returns:
        DictConfig object with merged configuration (supports attribute access)
    
    Examples:
        # Basic usage
        cfg = load_config()
        
        # With experiment
        cfg = load_config(overrides=["experiment=continue_att_backbone"])
        
        # Multiple overrides
        cfg = load_config(overrides=[
            "experiment=continue_att_backbone",
            "NUM_GPUS=4",
            "TRAIN.BATCH_SIZE=8"
        ])
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "configs")
    
    # Ensure absolute path
    config_path = str(Path(config_path).absolute())
    
    overrides = overrides or []
    
    # Clear any existing Hydra instance (allows multiple calls)
    GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
    
    # Create output directory if needed (matching original behavior)
    if create_output_dir and cfg.get("OUTPUT_DIR"):
        ckp.make_checkpoint_dir(cfg.OUTPUT_DIR)
    
    return cfg


def load_config_from_args(args=None) -> DictConfig:
    """
    Load config from command line arguments (legacy compatibility).
    
    Parses --cfg and opts from command line, similar to original Config class.
    
    Args:
        args: Optional argparse namespace. If None, parses sys.argv.
    
    Returns:
        DictConfig with merged configuration
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hydra config loader with legacy argument support"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="(Legacy) Path to the configuration file - use Hydra overrides instead",
        default=None
    )
    parser.add_argument(
        "--config-name",
        dest="config_name",
        help="Name of the main config (default: config)",
        default="config"
    )
    parser.add_argument(
        "--config-path",
        dest="config_path",
        help="Path to config directory",
        default=None
    )
    parser.add_argument(
        "opts",
        help="Override options in KEY=VALUE format",
        default=None,
        nargs=argparse.REMAINDER
    )
    
    parsed_args = parser.parse_args(args)
    
    # Convert legacy KEY VALUE format to KEY=VALUE
    overrides = []
    if parsed_args.opts:
        # Handle both "KEY VALUE" and "KEY=VALUE" formats
        i = 0
        while i < len(parsed_args.opts):
            opt = parsed_args.opts[i]
            if "=" in opt:
                overrides.append(opt)
                i += 1
            elif i + 1 < len(parsed_args.opts):
                # Legacy format: KEY VALUE
                overrides.append(f"{opt}={parsed_args.opts[i + 1]}")
                i += 2
            else:
                i += 1
        
    
    return load_config(
        config_name=parsed_args.config_name,
        overrides=overrides,
        config_path=parsed_args.config_path,
    )


def cfg_to_dict(cfg: DictConfig, resolve: bool = True) -> dict:
    """
    Convert OmegaConf DictConfig to regular dict.
    
    Args:
        cfg: The DictConfig to convert
        resolve: Whether to resolve interpolations
    
    Returns:
        Plain Python dict
    """
    return OmegaConf.to_container(cfg, resolve=resolve)


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """
    Pretty print the configuration.
    
    Args:
        cfg: The config to print
        resolve: Whether to resolve interpolations before printing
    """
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg, resolve=resolve))
    print("=" * 60 + "\n")


def print_config_sources(cfg: DictConfig) -> None:
    """
    Print information about where config values come from.
    
    Useful for debugging config composition issues.
    """
    print("\n" + "=" * 60)
    print("CONFIG STRUCTURE (after composition)")
    print("=" * 60)
    print("Composition order:")
    print("  1. base.yaml (defaults)")
    print("  2. model/*.yaml (architecture)")
    print("  3. training/*.yaml (hyperparameters)")
    print("  4. experiment/*.yaml (experiment-specific)")
    print("  5. CLI overrides")
    print("=" * 60)
    print("\nResolved configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60 + "\n")


def save_config(cfg: DictConfig, path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        cfg: The config to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        OmegaConf.save(cfg, f, resolve=True)


def get_config_diff(cfg1: DictConfig, cfg2: DictConfig) -> str:
    """
    Get a human-readable diff between two configs.
    
    Useful for comparing experiment configurations.
    """
    yaml1 = OmegaConf.to_yaml(cfg1, resolve=True)
    yaml2 = OmegaConf.to_yaml(cfg2, resolve=True)
    
    import difflib
    diff = difflib.unified_diff(
        yaml1.splitlines(keepends=True),
        yaml2.splitlines(keepends=True),
        fromfile="config1",
        tofile="config2",
    )
    return "".join(diff)


class ConfigWrapper:
    """
    Wrapper class for compatibility with code expecting the old Config class.
    
    Provides the same interface as the original Config class while using
    Hydra/OmegaConf under the hood.
    
    Usage:
        cfg = ConfigWrapper(load=True)
        # or
        cfg = ConfigWrapper(overrides=["experiment=continue_att_backbone"])
    """
    
    def __init__(
        self,
        load: bool = True,
        cfg_dict: Optional[DictConfig] = None,
        overrides: Optional[List[str]] = None,
        config_name: str = "config",
        config_path: Optional[str] = None,
    ):
        if load:
            if cfg_dict is not None:
                self._cfg = cfg_dict
            else:
                self._cfg = load_config(
                    config_name=config_name,
                    overrides=overrides,
                    config_path=config_path,
                )
            self.cfg_dict = cfg_to_dict(self._cfg)
        else:
            self._cfg = OmegaConf.create(cfg_dict or {})
            self.cfg_dict = cfg_dict or {}
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "cfg_dict":
            return super().__getattribute__(name)
        return getattr(self._cfg, name)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with a default."""
        return self._cfg.get(key, default)
    
    def __repr__(self) -> str:
        return f"ConfigWrapper(\n{OmegaConf.to_yaml(self._cfg)})"
    
    def dump(self) -> str:
        """Dump config as YAML string."""
        return OmegaConf.to_yaml(self._cfg, resolve=True)
    
    def deep_copy(self) -> "ConfigWrapper":
        """Create a deep copy of the config."""
        import copy
        return ConfigWrapper(load=False, cfg_dict=copy.deepcopy(self.cfg_dict))


# Alias for backward compatibility
Config = ConfigWrapper
