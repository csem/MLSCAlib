"""
MLSCAlib
Copyright (C) 2025 CSEM 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
#
# brief     : A custom Dataset class
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
import torch
from torch.utils.data import Dataset


class NRFDataset(Dataset):
    """ Class NRFDataset: is a class of type torch.utils.data.Dataset, and is used by the DataLoader in the training process
        to perform a batched learning.
    """
    def __init__(self, all_traces, all_labels,device = torch.device("cpu")):
        self.labels = all_labels
        self.traces = all_traces
        self.device = device
    def __len__(self):
        return len(self.traces)
    def __getitem__(self, idx):
        label = self.labels[idx]
        trace = self.traces[idx]
        return trace.to(self.device),label.to(self.device)

class NRFDataset3(Dataset):
    """ Class NRFDataset: is a class of type torch.utils.data.Dataset, and is used by the DataLoader in the training process
        to perform a batched learning.
    """
    def __init__(self, all_traces, all_labels,raw_traces,device = torch.device("cpu")):
        self.labels = all_labels
        self.traces = all_traces
        self.raw_traces = raw_traces
        self.device = device
    def __len__(self):
        return len(self.traces)
    def __getitem__(self, idx):
        label = self.labels[idx]
        trace = self.traces[idx]
        return trace.to(self.device),label.to(self.device),self.raw_traces[idx % len(self.raw_traces)].to(self.device)

class PredictionDataset(Dataset):
    """ Class PredictionDataset: is a class of type torch.utils.data.Dataset, and is used by the DataLoader in the prediction
         process to perform a batched prediction.
    """
    def __init__(self, all_traces,device = torch.device("cpu"),numpy=False):
        self.traces = all_traces
        self.device=device
        self.numpy=numpy
    def __len__(self):
        return len(self.traces)
    def __getitem__(self, idx):
        trace = self.traces[idx]
        if(self.numpy):
            return trace
        return trace.to(self.device)