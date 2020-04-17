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

        print(f'runing fold {fold_number}/{5}')

        model = config.ml_model(*config.model.args, **config.model.kwargs)
        model = mt.nn.init.XavierUniformWeightInitializer()(model)
        optimizer = torch.optim.Adam(model.parameters(), **config.optimizer.kwargs)
        criterion = torch.nn.L1Loss(**config.criterion)

        fit_callbacks = [
            mt.Callbacks.Tensorboard(),
            mt.Callbacks.LRFind(),
            mt.Callbacks.ReduceLROnValidPlateau(
                checkpoint=True, to_file=f'./fold_{fold_number}/valid/log',
                filename=f'./fold_{fold_number}/Checkpoints/checkpoint')
        ]

        test_callbacks = [
        ]

        trainer = mt.Trainer(model, optimizer, criterion, **config.trainer)

        start_time = time()

        trainer.fit_on_loader(
            train_loader, valid_dataloader=valid_loader, cycles=[100],
            callbacks=fit_callbacks, std_to_file=f'./fold_{fold_number}/train/log')

        end_time = time()

        train_time = end_time - start_time

        trainer.load_best_model(f'./fold_{fold_number}/Checkpoints')

        start_time = time()
        trainer.test_on_loader(
            test_loader, std_to_file=f'./fold_{fold_number}/test/log', callbacks=test_callbacks
        )
        end_time = time()

        test_time = end_time-start_time

        time_log.append(
            (train_time, test_time)
        )

        np.save('./time_log', time_log)

        del fold_number
        del model
        del optimizer
        del criterion
        del fit_callbacks
        del test_callbacks
        del trainer
        torch.cuda.empty_cache()
