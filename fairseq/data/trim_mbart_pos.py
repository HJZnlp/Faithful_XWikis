import torch
import os

#MODEL_DATASET = 'trimMLSUM'
MODEL_DATASET = 'trimXWikis'
#SIZE=514
SIZE=602
HOME = '/home/lperez/storage/pretrained'
MODEL = 'mbart50' #'mbart'
model = torch.load(os.path.join(HOME, "{}.{}/model.pt".format(MODEL,MODEL_DATASET)))

print(model["model"]["encoder.embed_positions.weight"].size())
print(model["model"]["decoder.embed_positions.weight"].size())

model["model"]["encoder.embed_positions.weight"] = model["model"]["encoder.embed_positions.weight"][:SIZE]
model["model"]["decoder.embed_positions.weight"] = model["model"]["decoder.embed_positions.weight"][:SIZE]
torch.save(model, os.path.join(HOME, "{}.{}/model_{}.pt".format(MODEL, MODEL_DATASET, SIZE-2)))
