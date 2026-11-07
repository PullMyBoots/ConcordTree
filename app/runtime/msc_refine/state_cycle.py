"""Brent recurrence detection for a deterministic full search state.

Holds one full-state checkpoint, not every previous adjacency. A recurrence
is an explicit nonconvergence failure; it is never a scientific success.
"""

class StateRecurrence:
    def __init__(self):
        self.checkpoint=None
        self.power=1
        self.distance=0

    def repeated(self,state):
        if self.checkpoint is None:
            self.checkpoint=state
            return False
        self.distance+=1
        if state==self.checkpoint:
            return True
        if self.distance==self.power:
            self.checkpoint=state
            self.power*=2
            self.distance=0
        return False
