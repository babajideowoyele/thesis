import enum
import pydantic

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