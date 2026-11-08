"""Publish complete trees atomically and pause only at valid tree boundaries."""
from pathlib import Path
import time

class Checkpoints:
    def __init__(self,out,pause_request=None):
        self.out=out
        self.pause_request=Path(pause_request) if pause_request else None
        self.paused_seconds=0.
        self.sequence=0

    def publish(self,tree,event=None):
        from concordtree.treeio import newick_python
        from engine import write_json
        tree.require_binary()
        temporary=self.out/'checkpoint.nwk.tmp'
        temporary.write_text(newick_python(tree)+'\n')
        temporary.replace(self.out/'checkpoint.nwk')
        self.sequence+=1
        write_json(self.out/'checkpoint.json',dict(available=True,sequence=self.sequence,taxa=len(tree.names),**(event or {})))
        if self.pause_request is not None and self.pause_request.exists():
            started=time.perf_counter()
            # Acknowledge each request token, including an immediate resume/pause.
            acknowledged=None
            while True:
                try:token=self.pause_request.read_text()
                except FileNotFoundError:break
                if token!=acknowledged:
                    write_json(self.out/'pause.state.json',dict(paused=True,sequence=self.sequence,request_token=token))
                    acknowledged=token
                time.sleep(.1)
            self.paused_seconds+=time.perf_counter()-started
            (self.out/'pause.state.json').unlink(missing_ok=True)
