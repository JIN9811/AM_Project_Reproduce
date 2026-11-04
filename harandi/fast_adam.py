"""Optional CUDA graph replay of an unchanged float64 Adam objective.

Every call performs exactly one update, including the eager warmup calls.
The closure must be static and return a tensor or a dict with a 'total' tensor.
Returned tensors are reused by later replays: copy scalars when logging.
See https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
"""
import torch


def index_batch(batch):
    """Avoid dynamic-shape boolean indexing inside a captured CUDA graph."""
    return {k:torch.nonzero(v,as_tuple=True)[0] if v.dtype==torch.bool else v for k,v in batch.items()}


class CapturedAdam:
    def __init__(self,optimizer,closure,warmup=3):
        if warmup<1:raise ValueError('warmup must include at least one actual update')
        if not all(p.is_cuda for g in optimizer.param_groups for p in g['params']):
            raise ValueError('CapturedAdam needs CUDA parameters')
        if not all(g.get('capturable') for g in optimizer.param_groups):
            raise ValueError('Adam must use capturable=True')
        self.optimizer=optimizer;self.closure=closure;self.warmup=warmup
        self.updates=0;self.graph=None;self.output=None;self.stream=torch.cuda.Stream()

    @property
    def captured(self):return self.graph is not None

    def _update(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.output=self.closure()
        loss=self.output['total'] if isinstance(self.output,dict) else self.output
        loss.backward();self.optimizer.step()

    def step(self):
        if self.updates<self.warmup:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):self._update()
            torch.cuda.current_stream().wait_stream(self.stream)
        else:
            if self.graph is None:
                torch.cuda.synchronize()
                self.graph=torch.cuda.CUDAGraph()
                # Capture records CUDA operations; replay below executes them.
                with torch.cuda.graph(self.graph,stream=self.stream):self._update()
            self.graph.replay()
        self.updates+=1
        return self.output
