from pathlib import Path
import myTorch as mt
import torch

input_name = 'curvature'
target_name = 'wss'

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        base = 600

        self.lin1 = torch.nn.Linear(4, base, bias=False)
        self.lin2 = torch.nn.Linear(base, base * 2, bias=False)
        self.lin3 = torch.nn.Linear(base * 2, base * 4, bias=False)
        self.lin4 = torch.nn.Linear(base * 4, base * 2, bias=False)
        self.lin5 = torch.nn.Linear(base * 2, base * 1, bias=False)
        self.lin6 = torch.nn.Linear(base * 1, base // 2, bias=False)
        self.lin7 = torch.nn.Linear(base // 2, 1, bias=False)

    def forward(self, x, *args, **kwargs):
        x = x.transpose(1,-1)
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.lin2(x)
        x = torch.nn.functional.relu(x)
        x = self.lin3(x)
        x = torch.nn.functional.relu(x)
        x = self.lin4(x)
        x = torch.nn.functional.relu(x)
        x = self.lin5(x)
        x = torch.nn.functional.relu(x)
        x = self.lin6(x)
        x = torch.nn.functional.relu(x)
        x = self.lin7(x)
        x = torch.nn.functional.relu(x)
        return x.transpose(1,-1)

ml_model = Model
current_directory = Path(__file__).parent

data_folder = current_directory/'../../../Data/LADSSMNewtonianSteadyWSS'
TVS = mt.Utils.AttrDict()
TVS.init = dict(input_name=input_name, target_name=target_name, RGB=True)
TVS.split = mt.Utils.AttrDict()
TVS.split.args = (0.1, )
TVS.split.kwargs = dict(random_state=1, n_splits=5)
TVS.call = dict(batch_size=1, shuffle=True)

model = mt.Utils.AttrDict()
model.args = (4, 1, 3)
model.kwargs = dict(n_pool=4, base_channels=64, output_activation=None)

optimizer = mt.Utils.AttrDict()
optimizer.kwargs = dict(lr=1e-7, eps=1e-7)

criterion = dict(reduction='sum')

trainer = dict(
    x_key=input_name, y_key=target_name
)
