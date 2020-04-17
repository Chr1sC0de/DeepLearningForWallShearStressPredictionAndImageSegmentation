from . import CallOn


class LRFind(CallOn):

    def __init__(
            self, on_batch=False, on_epoch=False, on_cycle=True,
            on_step=False, on_end=False, in_memory=False, to_file=None,
            on_startup=True, **lr_find_kwargs):
        # initialize when to save state
        super(LRFind, self).__init__(
            on_batch=on_batch, on_epoch=on_epoch, on_cycle=on_cycle,
            on_step=on_step, on_end=on_end
        )
        self.on_startup = on_startup
        self.lr_find_kwargs = lr_find_kwargs

    def pre_loop(self, env):
        if self.on_startup:
            env.lr_find(env.train_dataloader, **self.lr_find_kwargs)

    def method(self, env):
        del env.lr_finder
        env.optimizer.param_groups[0]['lr'] = \
            env.optimizer.param_groups[0]['lr']/100
        env.lr_find(env.train_dataloader, **self.lr_find_kwargs)
