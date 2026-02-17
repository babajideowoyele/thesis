import os
from src import get_model_class
import torch

class MVFoulModel(torch.nn.Module):
    def __init__(self, config):
        super(MVFoulModel, self).__init__()
        self.backbone = get_model_class(config.model.backbone.name)(num_frames=config.model.backbone.num_frames)
        self._load_pretrained_weights(config.model.backbone.pretrained_weights_path)
        self.classifier_action = get_model_class(config.model.classifier_action.name)(config.model.classifier_action)
        self.classifier_offence_severity = get_model_class(config.model.classifier_severity.name)(config.model.classifier_severity)
        self.grid_size = 16

        if config.model.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if x.ndim == 6:
            # If the input has 6 dimensions, reshape to (B*V, C, T, H, W) for backbone
            b, v, c, t, h, w = x.shape
            x = x.view(b * v, c, t, h, w)
            num_views = v
        else:
            b, c, t, h, w = x.shape
            num_views = 1
        
        features = self.backbone(x)  # Shape: (B*V, num_patches, embed_dim)
        
        # Reshape back to (B, V, num_patches, embed_dim) then pool each view
            # Average pool across views: (B, V, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        num_features = (features.shape[1] // self.grid_size*self.grid_size) * self.grid_size*self.grid_size
        features = features[:, num_features:]
        action_pred = self.classifier_action(features) if self.classifier_action is not None else None
        offence_severity_pred = self.classifier_offence_severity(features) if self.classifier_offence_severity is not None else None
        if num_views > 1:
            action_pred = action_pred.view(b, v, -1).mean(dim=1) if action_pred is not None else None
            offence_severity_pred = offence_severity_pred.view(b, v, -1).mean(dim=1) if offence_severity_pred is not None else None
        
        return action_pred, offence_severity_pred
    
    def _load_pretrained_weights(self, model_path: str):
        """Load pretrained VJEPA weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        pretrained_dict = torch.load(model_path, weights_only=True, map_location="cpu")["encoder"]
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = self.backbone.load_state_dict(pretrained_dict, strict=False)
        print(f"Loaded pretrained weights from {model_path}")
        print(f"Load message: {msg}")
    
def get_model(config):
    model = MVFoulModel(config)
    return model
    
