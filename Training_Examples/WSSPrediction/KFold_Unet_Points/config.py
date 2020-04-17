from pathlib import Path
import myTorch as mt

input_name = 'points'
target_name = 'wss'

ml_model = mt.Models.Unet

current_directory = Path(__file__).parent

data_folder = current_directory/'../../../Data/NewtonianSteadyWSS'
TVS = mt.Utils.AttrDict()
TVS.init = dict(input_name=input_name, target_name=target_name)
TVS.split = mt.Utils.AttrDict()
TVS.split.args = (0.1, )
TVS.split.kwargs = dict(random_state=1, n_splits=5)
TVS.call = dict(batch_size=1, shuffle=True)

model = mt.Utils.AttrDict()
model.args = (3, 1, 3)
model.kwargs = dict(n_pool=4, base_channels=64, output_activation=None)

optimizer = mt.Utils.AttrDict()
optimizer.kwargs = dict(lr=1e-7, eps=1e-7)

criterion = dict(reduction='sum')

trainer = dict(
    x_key=input_name, y_key=target_name
)
