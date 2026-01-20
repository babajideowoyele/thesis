

def train_epoch(model, rank, cfg, data):
    model.train()
    for (feature, mask), label in data:
        pred = model(feature, mask)

        loss = 
