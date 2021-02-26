import torch as _torch


class ToGPU:
    device = "cuda:0"

    def __call__(self, sample_dict):
        self.device = _torch.device(
            self.device if _torch.cuda.is_available else "cpu"
        )

        for key in sample_dict.keys():
            try:
                sample_dict[key] = sample_dict[key].to(self.device)
            except ValueError:
                print(f'{key} of {sample_dict[key].__call__} could not be placed on GPU')

        return sample_dict
