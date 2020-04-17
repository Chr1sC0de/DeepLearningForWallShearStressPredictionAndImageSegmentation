import torch
import pathlib as pt


class ModelInitializer:

    def __init__(self, initializer, *args, **kwargs):
        self.initializer = initializer
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model):
        for param in model.parameters():
            if self.criterion(param):
                self.initializer(param, *self.args, **self.kwargs)
            return model

    def criterion(self, param):
        return True


class WeightInitializer(ModelInitializer):
    def criterion(self, param):
        if len(param.shape) == 4:
            return True
        return False


class XavierUniformWeightInitializer(WeightInitializer):
    def __init__(self, gain=1.0):
        super(XavierUniformWeightInitializer, self).__init__(
            torch.nn.init.xavier_uniform_,
            gain=gain
        )


class LoadBest:

    def __init__(self, folder, score='validation_score', model_name='model'):
        self.checkpoint_folder = pt.Path(folder)
        files = list(self.checkpoint_folder.glob('*'))
        self.checkpoints = [torch.load(file) for file in files]
        if score is not None:
            self.checkpoints.sort(key=lambda x: x[score])
        self.checkpoint_id = -1
        self.model_name = model_name

    def __call__(self, model, checkpoint_id=None):
        if checkpoint_id is not None:
            self.checkpoint_id = checkpoint_id
        model.load_state_dict(
            self.checkpoints[self.checkpoint_id][self.model_name]
        )
        return model




