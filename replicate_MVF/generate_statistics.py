from tadaconv.datasets.base.mvfoul import Mvfoul
from tadaconv.utils.config import Config
from tadaconv.utils.mvfoul_translation import ActionClass

cfg = Config()
mvfoul = Mvfoul(cfg, split="train")

labels = mvfoul.labels

label_counts = {
    k: 0 for k in range(9)}

severity_counts = {
    k: 0 for k in range(5)}

for label in labels:
    action_class = label['type']
    severity = label['severity']
    label_counts[action_class] += 1
    severity_counts[severity] += 1

for k in range(9):
    print(f"Action Class {k} ({ActionClass(k).name}): {label_counts[k]} ({label_counts[k]/len(labels)}) occurrences")

for k in range(5):
    print(f"Severity {k}: {severity_counts[k]} ({severity_counts[k]/len(labels)}) occurrences")