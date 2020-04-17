import myTorch as mt
import torch
import config
import numpy as np
from time import time



if __name__ == "__main__":
    numpy_files = mt.Utils.glob_npz(config.data_folder)

    transform = [mt.Data.Transforms.FlipTensor()]

    KFoldTVSLoader = mt.Data.DataLoaders.KFoldCrossTrainTestSplit(
       numpy_files, transform=transform, **config.TVS.init)

    KFoldTVSLoader.split(*config.TVS.split.args, **config.TVS.split.kwargs)

    split = KFoldTVSLoader(**config.TVS.call)

    time_log = []

    for fold_number, (train_loader, valid_loader, test_loader) in enumerate(split):

        if fold_number == 0:
            time_log.append(
                dict(
                    train_steps_per_epoch=len(train_loader),
                    test_steps_per_epoch=len(test_loader)
                )
            )

        fold_number += 1

        if fold_number in [1]:

            print(f'runing fold {fold_number}/{5}')

            model = config.ml_model(*config.model.args, **config.model.kwargs)
            criterion = torch.nn.L1Loss(**config.criterion)

            test_callbacks = [
            ]

            trainer = mt.Trainer(model, None, criterion, **config.trainer)

            trainer.load_best_model(f'./fold_{fold_number}/Checkpoints')

            start_time = time()
            trainer.test_on_loader(
                test_loader, std_to_file=f'./fold_{fold_number}/test/log', callbacks=test_callbacks
            )
            end_time = time()

            test_time = end_time-start_time

            del fold_number
            del model
            del criterion
            del test_callbacks
            del trainer
            torch.cuda.empty_cache()
