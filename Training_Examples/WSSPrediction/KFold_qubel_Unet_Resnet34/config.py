from pathlib import Path
import myTorch as mt
import segmentation_models_pytorch as smp
from torch import nn

input_name = 'curvature'
target_name = 'wss'


def model_maker(*args, **kwargs):
    return nn.Sequential(
        mt.nn.ConvNormAct2d(4, 3, 3),
        smp.Unet('resnet34', classes=1, activation=None)
    )


ml_model = model_maker

current_directory = Path(__file__).parent

data_folder = current_directory/'../../../Data/LADSSMNewtonianSteadyWSS'
TVS = mt.Utils.AttrDict()
TVS.init = dict(input_name=input_name, target_name=target_name)
TVS.split = mt.Utils.AttrDict()
TVS.split.args = (0.1, )
TVS.split.kwargs = dict(random_state=1, n_splits=5)
TVS.call = dict(batch_size=8, shuffle=True)

model = mt.Utils.AttrDict()
model.args = (4, 1, 3)
model.kwargs = dict(base_channels=64, output_activation=None)

optimizer = mt.Utils.AttrDict()
optimizer.kwargs = dict(lr=1e-7, eps=1e-7)

criterion = dict(reduction='sum')

trainer = dict(
    x_key=input_name, y_key=target_name
)
