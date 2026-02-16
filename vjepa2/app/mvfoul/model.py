from src import get_model_class

class MVFoulModel:
    def __init__(self, config):
        self.backbone = get_model_class(config.model.backbone)
        self.classifier_action = get_model_class(config.model.classifier_action)
        self.classifier_offence_severity = get_model_class(config.model.classifier_offence_severity)
    
    def forward(self, x):
        features = self.backbone(x)
        action_pred = self.classifier_action(features) if self.classifier_action is not None else None
        offence_severity_pred = self.classifier_offence_severity(features) if self.classifier_offence_severity is not None else None
        return action_pred, offence_severity_pred
    
def get_model(config):
    model = MVFoulModel(config)
    return model
    
