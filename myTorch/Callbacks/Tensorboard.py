from . import Callback
from torch.utils.tensorboard import SummaryWriter


class Tensorboard(Callback):

    def __init__(self, **kwargs):
        self.writer = SummaryWriter(**kwargs)
        self.graph_constructed = False

    def post_batch(self, environment):
        if not self.graph_constructed:
            self.writer.add_graph(
                environment.model, environment.x.float()
            )
            self.graph_constructed = True

    def post_loop(self, environement):
        self.writer.close()
