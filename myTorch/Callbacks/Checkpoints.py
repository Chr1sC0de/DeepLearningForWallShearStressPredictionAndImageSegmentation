from . import CallOn
import torch
from pathlib import Path


class Checkpoints(CallOn):
    def __init__(
        self, on_batch=False, on_epoch=False, on_cycle=False, on_step=False,
        on_end=True, filename='./Checkpoints/checkpoint', to_keep=10
    ):
        # initialize when to save state
        super(Checkpoints, self).__init__(
            on_batch=on_batch, on_epoch=on_epoch, on_cycle=on_cycle,
            on_step=on_step, on_end=on_end
        )
        # initialize the directory path and the directory
        # name
        self.directory = Path(filename).parent
        self.filename = Path(filename).name
        self.to_keep = to_keep

    def pre_loop(self, env):
        if not self.directory.exists():
            self.directory.mkdir()

    def method(self, env):
        self.save(env)

    def save(self, env):
        checkpoints = list(self.directory.glob(
            '%s*' % self.filename
        ))
        if len(checkpoints) > self.to_keep:
            checkpoints[0].unlink()

        latest_checkpoint = "%s_%05d" % (self.filename, -1)
        for checkpoint in checkpoints:
            latest_checkpoint = checkpoint.name
        checkpoint_number = latest_checkpoint.split('_')[-1]
        checkpoint_number = int(checkpoint_number)
        checkpoint_name = "%s_%05d" % (self.filename, checkpoint_number + 1)
        torch.save(
            self.dict_constructor(env),
            self.directory/checkpoint_name
        )
        return self.directory/checkpoint_name

    @staticmethod
    def dict_constructor(env):
        return dict(
                model=env.model.state_dict(),
                optimizer=env.optimizer.state_dict(),
                loss_score=env.loss,
                accuracy_score=env.accuracy
            )
