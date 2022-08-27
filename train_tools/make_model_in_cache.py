import sys
import torch

sys.path.insert(0, "../")

from models.Model import get_model, slice_model

torch.save(get_model("cuda:0").getSliceModel().eval().state_dict(), "../ModelCache/model_1.pt")