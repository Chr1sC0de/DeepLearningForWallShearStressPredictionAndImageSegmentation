class Callback:
    def __init__(self):
        pass

    def pre_loop(self, environment):
        pass

    def pre_cycle(self, environment):
        pass

    def pre_epoch(self, environment):
        pass

    def pre_batch(self, environment):
        pass

    def post_batch(self, environment):
        pass

    def post_epoch(self, environment):
        pass

    def post_cycle(self, environment):
        pass

    def post_loop(self, environment):
        pass


class CallOn(Callback):

    def __init__(
        self, on_batch=False, on_epoch=False, on_cycle=False,
            on_step=False, on_end=True):

        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.on_cycle = on_cycle
        self.on_step = on_step
        self.on_end = on_end
        self.global_step = 0

    def post_batch(self, env):
        if self.on_step:
            if self.global_step % self.on_step == 0:
                self.method(env)
        if self.on_batch:
            if env.i_batch % self.on_batch == 0:
                self.method(env)
        self.global_step += 1

    def post_epoch(self, env):
        if self.on_epoch:
            if env.i_epoch % self.on_epoch == 0:
                self.method(env)

    def post_cycle(self, env):
        if self.on_cycle:
            if env.i_cycle % self.on_cycle == 0:
                self.method(env)

    def post_loop(self, env):
        if self.on_end:
            self.method(env)

    def method(self, env):
        NotImplemented
