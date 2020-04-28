import myTorch as mt
import torch
import config
import numpy as np
from time import time

class RolledTester(mt.Trainer):

    def test_step(self, data_dict):

        self.run_extractor_hooks(data_dict)
        self.x, self.y_true = self.data_extractor(data_dict)
        self.x = self.x.to(self.device)
        self.y_true = self.y_true.to(self.device)
        roll_values = torch.linspace(0, self.x.shape[2], 10)
        predictions = []
        for val in roll_values:
            rolled_tensor = torch.roll(self.x, int(val), dims=2)
            predictions.append(self.model(rolled_tensor))
        predictions = [
            torch.roll(p, -int(val), dims=2) for p, val in zip(predictions, roll_values)
        ]
        self.y_pred = torch.mean(torch.stack(predictions, dim=0), dim=0)
        self.loss = self.criterion(self.y_pred, self.y_true)
        self.accuracy = self.get_accuracy()

if __name__ == "__main__":
    numpy_files = mt.Utils.glob_npz(config.data_folder)

    test_loader = torch.utils.data.DataLoader(
        mt.Data.DataLoaders.DictDataset(
            numpy_files, input_name='curvature', target_name='wss'))

    time_log = []

    time_log.append(
        dict(
            test_steps_per_epoch=len(test_loader)
        )
    )

    model = config.ml_model(*config.model.args, **config.model.kwargs)

    test_callbacks = [
        mt.Callbacks.SavePyvistaPoints(
            on_batch=1, properties=['curvature'],
            filename=f'./test/data')
    ]

    criterion = torch.nn.L1Loss(**config.criterion)

    trainer = RolledTester(model, None, criterion, **config.trainer)

    trainer.load_best_model(f'D:/Github/myTorch/Training_Examples/WSSPrediction/KFold_CSESEPN_U/fold_1/Checkpoints')

    start_time = time()
    trainer.test_on_loader(
        test_loader, std_to_file=f'./test/log', callbacks=test_callbacks
    )
    end_time = time()

    test_time = end_time-start_time

    time_log.append(
        (test_time)
    )

    np.save('./time_log', time_log)

    del model
    del criterion
    del test_callbacks
    del trainer
    torch.cuda.empty_cache()
