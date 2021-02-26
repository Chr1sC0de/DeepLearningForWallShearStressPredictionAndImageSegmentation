from .ToGPU import ToGPU


class TransformSequence:

    def __init__(self, *args, auto_gpu=True):
        self.transform_list = []
        self.transform_list.extend(args)
        if auto_gpu:
            self.transform_list.extend([ToGPU()])

    def __len__(self):
        return len(self.transform_list)

    def __call__(self, sample_list):
        for transform in self.transform_list:
            sample_list = transform(sample_list)
            assert sample_list is not None
        return sample_list
