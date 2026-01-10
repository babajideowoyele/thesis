from tadaconv.models.base.builder import build_model
from tadaconv.utils.config import Config
import weightwatcher as ww
import tadaconv.utils.checkpoint as cu


cfg = Config(load=True)

model, model_ema = build_model(cfg, 0)

cu.load_train_checkpoint(cfg, model, model_ema, None, None)


watcher = ww.WeightWatcher(model=model)
details = watcher.analyze()
details.to_csv('details.csv', index=False)
print(details)
summary = watcher.get_summary(details)
print(summary)
