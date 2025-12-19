import enum
import pydantic
import torch

class ActionClass(enum.IntEnum):
    holding = 0
    tackling = 1
    standingtackling = 2
    elbowing = 3
    pushing = 4
    challenge = 5
    dive = 6
    highleg = 7
    noaction = 8

    @classmethod
    def create(cls, action_str: str):
        action_str = ''.join(action_str.lower().split())
        if action_str not in cls.__members__:
            return cls.noaction
        return cls[action_str]


class MVFoulAnnotation(pydantic.BaseModel):
    Offence: str
    Contact: str
    Bodypart: str
    UpperBP: str
    Action: ActionClass
    Severity: int
    MultipleFouls: str
    TryPlay: bool
    TouchBall: bool
    Handball: bool
    HandballOffence: str

    @classmethod
    def from_dict(cls, data: dict):
        severity_str = data["Severity"].lower()
        severity = int(float(severity_str)) if severity_str in ["1.0", "2.0", "3.0"] else 0
        
        return cls(
            Offence=data["Offence"],
            Contact=data["Contact"],
            Bodypart=data["Bodypart"],
            UpperBP=data["Upper body part"],
            Action=ActionClass.create(data["Action class"]),
            Severity=severity,
            MultipleFouls=data["Multiple fouls"],
            TryPlay=data["Try to play"] == "Yes",
            TouchBall=data["Touch ball"] == "Yes",
            Handball=data["Handball"] != "No handball",
            HandballOffence=data["Handball offence"]
        )


def translate_annotation(annotation):
    foul_annotation = MVFoulAnnotation.from_dict(annotation)
    translated = {
        "type": foul_annotation.Action.value,
        "severity": foul_annotation.Severity,
    }
    return translated

def report_bad_examples(cfg, preds, labels):
    threshold = cfg.TRAIN.BAD_EXAMPLE_THRESHOLD if cfg.TRAIN.BAD_EXAMPLE_THRESHOLD is not None else 0.
    bad_examples = []

    w_type = cfg.DATA.CLASS_WEIGHTS.TYPE
    w_sev = cfg.DATA.CLASS_WEIGHTS.SEVERITY

    p_type = preds["type"]      # (N, C_t), probs
    p_sev = preds["severity"]   # (N, C_s), probs
    y_type = labels['supervised']["type"]     # (N,)
    y_sev = labels['supervised']["severity"]  # (N,)
    for i in range(p_type.shape[0]):
        pt_i = p_type[i]                    # (C_t,)
        yt_i = y_type[i].item()
        # max prob over wrong classes
        mask_wrong_t = torch.ones_like(pt_i, dtype=torch.bool)
        mask_wrong_t[yt_i] = False
        worst_conf_t = pt_i[mask_wrong_t].max().item()/torch.abs(pt_i[yt_i]).item()

        ps_i = p_sev[i]                    # (C_t,)
        ys_i = y_sev[i].item()
        # max prob over wrong classes
        mask_wrong_s = torch.ones_like(ps_i, dtype=torch.bool)
        mask_wrong_s[ys_i] = False
        worst_conf_s = ps_i[mask_wrong_s].max().item()/torch.abs(ps_i[ys_i]).item()

        score = w_type * worst_conf_t + w_sev * worst_conf_s

        if score > threshold:
            bad_examples.append({
                "meta_data": labels['meta_data']['dir_name'][i],
                "true_type": ActionClass(yt_i),
                "pred_type": ActionClass(pt_i.argmax().item()),
                "true_severity": ys_i,
                "pred_severity": ps_i.argmax().item(),
                "score": score,
            })

    bad_examples.sort(key=lambda x: x["score"], reverse=True)
    return bad_examples if bad_examples else None