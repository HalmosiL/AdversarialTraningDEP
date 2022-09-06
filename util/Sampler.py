import torch
import sys

class Iterable:
  def __init__(self):
    self.index_ =  0
    self.backQueueidx_ = []
    
  def __getitem__(self, idx):
    if(len(self.backQueueidx_) == 0):
      index_ = self.index_
      self.index_ += 1
    else
      index_ = self.backQueueidx_[0]
      self.backQueueidx_.pop(0)
      
    return index_

  def __len__(self):
      return sys.maxint
    
  def __back__(self, idx_):
    self.backQueue.append(idx_)

class FileSampler(torch.utils.data.Sampler):
  def __init__(self):
      self.iterable = Iterable()

  def __iter__(self):
      return iter(self.iterable)

  def __len__(self):
      return len(self.iterable)
