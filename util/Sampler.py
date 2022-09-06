import torch
import sys

class Iterable:
  def __init__(self):
    self.index_ =  0
    
  def __getitem__(self, idx):
    index_ = self.index_
    self.index_ += 1
    return index_

  def __len__(self):
      return sys.maxint
    
  def __back__(self):
    self.index_ += 1

class FileSampler(torch.utils.data.Sampler):
  def __init__(self):
      self.iterable = Iterable()

  def __iter__(self):
      return iter(self.iterable)

  def __len__(self):
      return len(self.iterable)
