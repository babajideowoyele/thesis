"""
Config-driven metric factory for the multi-task MVFoul evaluator.

The evaluator engine is expected to return a dict with keys:
    offence_sev_pred, offence_sev_target, action_pred, action_target

Hydra config example (inside train_scratch.yaml):

    metrics:
      primary: BalancedAccuracy        # name used for best-model checkpointing
      eval:
        SeverityBalancedAccuracy:
          type: balanced_accuracy
          task: severity
        SeverityAccuracy:
          type: accuracy
          task: severity
        ActionAccuracy:
          type: accuracy
          task: action
        ActionBalancedAccuracy:
          type: balanced_accuracy
          task: action
        Loss:
          type: loss
          task: combined
"""

from typing import Dict, Tuple, Callable
import torch
from omegaconf import DictConfig
from ignite.metrics import Accuracy, Loss, Recall, Metric, ConfusionMatrix


# ---------------------------------------------------------------------------
# Output-transform helpers
# ---------------------------------------------------------------------------

def _severity_output_transform(output: dict):
    """Return (pred_logits, target_indices) for severity head."""
    return (
        output["offence_sev_pred"],
        torch.argmax(output["offence_sev_target"], dim=1),
    )


def _action_output_transform(output: dict):
    """Return (pred_logits, target_indices) for action head."""
    return (
        output["action_pred"],
        torch.argmax(output["action_target"], dim=1),
    )


def _severity_onehot_output_transform(output: dict):
    """Return (pred_logits, target_indices) for severity."""
    return output["offence_sev_pred"], torch.argmax(output["offence_sev_target"], dim=1)


def _action_onehot_output_transform(output: dict):
    """Return (pred_logits, target_indices) for action."""
    return output["action_pred"], torch.argmax(output["action_target"], dim=1)


def _combined_loss_output_transform(output: dict):
    """Pack both heads into a tuple pair for the combined loss metric."""
    return (
        (output["offence_sev_pred"], output["action_pred"]),
        (output["offence_sev_target"], output["action_target"]),
    )


def _severity_loss_output_transform(output: dict):
    return output["offence_sev_pred"], output["offence_sev_target"]


def _action_loss_output_transform(output: dict):
    return output["action_pred"], output["action_target"]


# ---------------------------------------------------------------------------
# Builders for individual metric types
# ---------------------------------------------------------------------------

_TASK_OUTPUT_TRANSFORMS = {
    "severity": _severity_output_transform,
    "action": _action_output_transform,
}

_TASK_ONEHOT_OUTPUT_TRANSFORMS = {
    "severity": _severity_onehot_output_transform,
    "action": _action_onehot_output_transform,
}

_TASK_LOSS_OUTPUT_TRANSFORMS = {
    "severity": _severity_loss_output_transform,
    "action": _action_loss_output_transform,
    "combined": _combined_loss_output_transform,
}


def _build_accuracy(task: str, **kwargs) -> Metric:
    return Accuracy(output_transform=_TASK_OUTPUT_TRANSFORMS[task])


def _build_balanced_accuracy(task: str, num_classes: int = None, **kwargs) -> Metric:
    """
    Macro-averaged recall == balanced accuracy.
    For multiclass, Ignite's Recall needs (logits, target_indices) and is_multilabel=False.
    """
    if num_classes is None:
        raise ValueError("balanced_accuracy metric requires 'num_classes' to be set in config (or inferred).")
    return Recall(
        average=True,
        is_multilabel=False,
        output_transform=_TASK_ONEHOT_OUTPUT_TRANSFORMS[task],
    )


def _build_loss(
    task: str,
    criterion_severity=None,
    criterion_action=None,
    **kwargs,
) -> Metric:
    if task == "combined":
        def combined_loss_fn(y_pred, y):
            sev_pred, act_pred = y_pred
            sev_target, act_target = y
            return criterion_severity(sev_pred, sev_target) + criterion_action(act_pred, act_target)

        return Loss(
            combined_loss_fn,
            output_transform=_combined_loss_output_transform,
            skip_unrolling=True,
        )
    elif task == "severity":
        return Loss(criterion_severity, output_transform=_severity_loss_output_transform)
    elif task == "action":
        return Loss(criterion_action, output_transform=_action_loss_output_transform)
    else:
        raise ValueError(f"Unknown loss task: {task}")


def _build_confusion_matrix(task: str, num_classes: int = None, **kwargs) -> Metric:
    """
    Build a confusion matrix for the specified task.
    The diagonal represents correct predictions, off-diagonal are errors.
    """
    if num_classes is None:
        raise ValueError("confusion_matrix metric requires 'num_classes' to be set in config.")
    return ConfusionMatrix(
        num_classes=num_classes,
        output_transform=_TASK_OUTPUT_TRANSFORMS[task],
    )


_METRIC_BUILDERS = {
    "accuracy": _build_accuracy,
    "balanced_accuracy": _build_balanced_accuracy,
    "loss": _build_loss,
    "confusion_matrix": _build_confusion_matrix,
}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_metrics(
    cfg: DictConfig,
    criterion_severity,
    criterion_action,
) -> Tuple[Dict[str, Metric], str]:
    """
    Build a dict of Ignite metrics from the ``metrics`` config section.

    Returns:
        metrics: ``{name: Metric}`` dict to attach to an evaluator.
        primary_metric: the name of the metric used for best-model checkpointing.
    """
    metrics_cfg = cfg.metrics
    primary = metrics_cfg.primary

    metrics: Dict[str, Metric] = {}
    for name, m_cfg in metrics_cfg.eval.items():
        mtype = m_cfg.type
        task = m_cfg.get("task", "severity")
        builder = _METRIC_BUILDERS.get(mtype)
        if builder is None:
            raise ValueError(
                f"Unknown metric type '{mtype}'. Available: {list(_METRIC_BUILDERS.keys())}"
            )
        # Forward extra kwargs (e.g. num_classes) + the criterion references
        extra = {k: v for k, v in m_cfg.items() if k not in ("type", "task")}
        metrics[name] = builder(
            task=task,
            criterion_severity=criterion_severity,
            criterion_action=criterion_action,
            **extra,
        )

    if primary not in metrics:
        raise ValueError(
            f"Primary metric '{primary}' not found in eval metrics {list(metrics.keys())}"
        )

    return metrics, primary
