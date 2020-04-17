import myTorch as mt
import torch
import config

if __name__ == "__main__":
    numpy_files = mt.Utils.glob_numpy(config.data_folder)

    transform = [mt.Data.Transforms.RollTensor(), ]

    TVSLoader = mt.Data.DataLoaders.TrainValidationTestSplit(
       numpy_files, transform=transform, **config.TVS.init)
    TVSLoader.split(
       *config.TVS.split.args, **config.TVS.split.kwargs)
    train_loader, valid_loader, test_loader = TVSLoader(**config.TVS.call)

    model = mt.Models.Unet(*config.model.args, **config.model.kwargs)
    model = mt.nn.init.XavierUniformWeightInitializer()(model)
    optimizer = torch.optim.Adam(model.parameters(), **config.optimizer.kwargs)
    criterion = torch.nn.L1Loss(**config.criterion)

    fit_callbacks = [
        mt.Callbacks.Tensorboard(),
        mt.Callbacks.Validation(checkpoint=True, on_step=100, to_file='./valid/log'),
        mt.Callbacks.SavePyvistaPoints(
            on_step=1000, properties=['curvature'],
            filename='./train/data'),
        mt.Callbacks.LRFind(),
        mt.Callbacks.ReduceLROnValidPlateau()
    ]

    test_callbacks = [
        mt.Callbacks.SavePyvistaPoints(
            on_batch=1, properties=['curvature'],
            filename='./test/data')
    ]

    trainer = mt.Trainer(model, optimizer, criterion, **config.trainer)
    if False:
        trainer.fit_on_loader(
            train_loader, valid_dataloader=valid_loader, cycles=[300, ],
            callbacks=fit_callbacks, std_to_file='./train/log')
    trainer.load_best_model('./Checkpoints')
    trainer.test_on_loader(
        test_loader, std_to_file='./test/log', callbacks=test_callbacks
    )
