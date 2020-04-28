import myTorch as mt
import torch
import config
import numpy as np
from time import time


if __name__ == "__main__":
    numpy_files_data = mt.Utils.glob_npz(config.data_folder)
    numpy_files_true = mt.Utils.glob_npz(config.true_data)

    transform = [mt.Data.Transforms.RollTensor(), ]

    KFoldTVSLoader_Synthetic = mt.Data.DataLoaders.KFoldCrossTrainTestSplit(
       numpy_files_data, transform=transform, **config.TVS.init)

    KFoldTVSLoader_True = mt.Data.DataLoaders.KFoldCrossTrainTestSplit(
       numpy_files_true, transform=transform, **config.TVS.init)

    KFoldTVSLoader_Synthetic.split(*config.TVS.split.args, **config.TVS.split.kwargs)
    KFoldTVSLoader_True.split(*config.TVS.split.args, **config.TVS.split.kwargs)

    split_synthetic = KFoldTVSLoader_Synthetic(**config.TVS.call)
    split_true = KFoldTVSLoader_True(**config.TVS.call)

    time_log = []

    for fold_number, \
        ( (train_synthetic, valid_synthetic, test_synthetic),
            (train_true, valid_true, test_true) ) in \
                enumerate(zip(split_synthetic, split_true)):

        # compile the synthetic datasets
        train_synthetic.dataset.file_paths += test_synthetic.dataset.file_paths
        n_synthetic = len(train_synthetic)
        n_true = len(train_true)
        # combine the true data
        train_true.dataset.file_paths += valid_true.dataset.file_paths
        for _ in range(((n_synthetic)//2)//n_true):
            train_synthetic.dataset.file_paths += train_true.dataset.file_paths

        # now set which dataloaders will be the train, test and valid
        train_loader = train_synthetic
        test_loader = test_true
        # use the synthetic data for validation to avoid overfitting
        valid_loader = valid_synthetic

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
            mt.Callbacks.SavePyvistaPoints(
                on_step=1000, properties=['curvature'],
                filename=f'./fold_{fold_number}/train/data'),
            mt.Callbacks.LRFind(),
            mt.Callbacks.ReduceLROnValidPlateau(
                checkpoint=True, to_file=f'./fold_{fold_number}/valid/log',
                filename=f'./fold_{fold_number}/Checkpoints/checkpoint')
        ]

        test_callbacks = [
            mt.Callbacks.SavePyvistaPoints(
                on_batch=1, properties=['curvature'],
                filename=f'./fold_{fold_number}/test/data')
        ]

        trainer = mt.Trainer(model, optimizer, criterion, **config.trainer)

        start_time = time()

        trainer.fit_on_loader(
            train_loader, valid_dataloader=valid_loader, cycles=[50, 50],
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
