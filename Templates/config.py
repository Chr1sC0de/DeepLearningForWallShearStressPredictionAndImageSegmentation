from pathlib import Path
import myTorch as mt
import torch

input_name = 'curvature'
target_name = 'wss'


current_directory = Path(__file__).parent

data_folder = current_directory/r'..\__CURVATUREDATA\DATA'

TVS = mt.Utils.AttrDict()
TVS.init = dict(input_name=input_name, target_name=target_name)
TVS.split = mt.Utils.AttrDict()
TVS.split.args = (0.7, 0.1, 0.2)
TVS.split.kwargs = dict(random_state=1)
TVS.call = dict(batch_size=1, shuffle=False)

model = mt.Utils.AttrDict()
model.args = (3, 1, 3)
model.kwargs = dict(n_pool=4, base_channels=64, output_activation=torch.nn.functional.relu)

optimizer = mt.Utils.AttrDict()
optimizer.kwargs = dict(lr=1e-7, eps=1e-7)

criterion = dict(reduction='sum')

trainer = dict(
    x_key=input_name, y_key=target_name
)
