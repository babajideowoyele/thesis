from tadaconv.models.base.base_blocks import PREAGGREGATE_REGISTRY


@PREAGGREGATE_REGISTRY.register()
class IdentityPreaggregate:
    def __init__(self, cfg):
        pass

    def forward(self, x):
        return x