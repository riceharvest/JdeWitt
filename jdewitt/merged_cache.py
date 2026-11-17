# SPDX-License-Identifier: AGPL-3.0-or-later
"""One explicit same-device budget shared by all edited projections."""
from collections import OrderedDict
import torch


class MergedCache:
    def __init__(self,budget_bytes=0):
        if type(budget_bytes) is not int or budget_bytes<0:
            raise ValueError('merged cache budget must be a nonnegative integer byte count')
        self.budget_bytes=budget_bytes
        self.used_bytes=0
        self.entries=OrderedDict()
        self.hits=self.misses=0

    def discard(self,key):
        entry=self.entries.pop(key,None)
        if entry is not None:
            self.used_bytes-=entry[1].numel()*entry[1].element_size()

    def clear(self):
        self.entries.clear()
        self.used_bytes=0

    def weight(self,adapter,weight):
        tensors=(weight,adapter.A,adapter.B,adapter.row_scale)
        # Never cache a detached training graph. Inference tensors may have no
        # mutation counter: without that proof, recompute rather than reuse.
        if not self.budget_bytes or (torch.is_grad_enabled() and any(t.requires_grad for t in tensors)):
            return adapter.merge_source(weight)
        try:
            signature=tuple((id(t),t.data_ptr(),t._version,t.device,t.dtype) for t in tensors)
        except RuntimeError:
            return adapter.merge_source(weight)
        key=id(adapter)
        entry=self.entries.get(key)
        if entry is not None and entry[0]==signature:
            self.entries.move_to_end(key)
            self.hits+=1
            return entry[1]
        self.discard(key)
        self.misses+=1
        size=weight.numel()*weight.element_size()
        if size>self.budget_bytes:
            return adapter.merge_source(weight)
        # Pin admitted layers until edits change. LRU would churn through the
        # whole depth each token and get zero hits when only part of a model fits.
        if self.used_bytes+size>self.budget_bytes:
            return adapter.merge_source(weight)
        rounded=adapter.merge_source(weight).detach()
        self.entries[key]=(signature,rounded)
        self.used_bytes+=size
        return rounded
