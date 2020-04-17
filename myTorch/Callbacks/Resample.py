from .Callback import CallOn
import torch
import numpy as np


class ResampleTrainingData(CallOn):

    def __init__(self, on_batch=False, on_epoch=False, on_cycle=False,
            on_step=False, on_end=False, desired_accuracy=95):
        self.desired_accuracy = desired_accuracy
        super().__init__(
            on_batch=on_batch, on_epoch=on_epoch,
            on_cycle=on_cycle, on_step=on_step,
            on_end=on_end)

    def post_epoch(self, env):
        if self.on_epoch:
            if env.i_epoch > 0:
                if env.i_epoch % self.on_epoch == 0:
                    self.method(env)

    def method(self, env):
        dataset = env.train_dataloader.dataset
        N = len(dataset)

        sol_list = []

        with torch.no_grad():
            for i in range(N):
                x, y = env.data_extractor(dataset[i])
                if len(x.shape) ==3 :
                    x = x.unsqueeze(0)
                    y = y.unsqueeze(0)
                y_pred = env.model(x)
                error = torch.clamp(
                    2*torch.abs(y-y_pred)/(torch.abs(y+y_pred)+1e-5),
                    0,
                    1
                )
                accuracy = (100-error*100).mean().detach().to('cpu').numpy()
                sol_list.append( (dataset.file_paths[i], accuracy) )

        sol_list.sort(key=lambda x: x[1])
        accuracies = [item[1] for item in sol_list]

        mean = np.mean(accuracies)
        std = np.std(accuracies)

        clipoff_range = mean-std

        lowest_scorering_files = [
            item[0] for item in sol_list if
                all(
                    [
                        item[1] < clipoff_range,
                        item[1] < self.desired_accuracy
                    ]
                )
        ]

        env.train_dataloader.dataset.file_paths += lowest_scorering_files

