from . import CallOn
from pathlib import Path
from typing import Iterable
import pyvista as pv
import numpy as np

class SaveIntermediate(CallOn):
    """SaveIntermediate

    Base class used to save batch data at some intermediate step

    Args:
        CallOn ([type]): [description]
    """
    figure_defaults = {}

    def __init__(
            self, *args, filename='./Intermediate/data', **kwargs):
        super(SaveIntermediate, self).__init__(*args, **kwargs)
        self.directory = Path(filename).parent
        self.filename = Path(filename).name

    def pre_loop(self, env):
        if not self.directory.exists():
            self.directory.mkdir()

    def method(self, env):
        self.construct(env)
        self.save(env)

    def construct_to_save(self, env):
        NotImplemented

    def save_file(self, env):
        NotImplemented


class SavePyvistaPoints(SaveIntermediate):
    extension = 'vtk'

    def __init__(
            self, *args, properties,
            points_key='points', on_end=False, **kwargs):
        super(SavePyvistaPoints, self).__init__(*args, on_end=on_end, **kwargs)
        assert isinstance(properties, Iterable), 'properties must be iterable'
        self.properties = properties
        self.points_key = points_key
        self.meshes_to_save = []
        self.grids_to_save = []

    def method(self, env):
        self.construct_to_save(env)
        for mesh in self.meshes_to_save:
            self.save(mesh)
        for grid in self.grids_to_save:
            self.save_grid(grid)

    def construct_to_save(self, env):

        batch_points = env.batch[self.points_key].detach().to('cpu').numpy()

        all_meshes = []
        all_grids = []

        for i, points in enumerate(batch_points):

            mesh = pv.StructuredGrid(
                points[0], points[1], points[2])
            grid = {
                'points': points
            }

            for prop in self.properties:
                data_prop = env.batch[prop][i].detach().to('cpu').numpy()
                self.assign_prop_to_mesh(mesh, prop, data_prop)
                grid[prop] = data_prop

            for prop, name in zip([env.y_true, env.y_pred], ['y_true', 'y_pred']):
                prop = prop.detach().to('cpu').numpy()[i]
                self.assign_prop_to_mesh(mesh, name, prop)
                grid[name] = prop

            all_meshes.append(mesh)
            all_grids.append(grid)

        self.meshes_to_save = all_meshes
        self.grids_to_save = all_grids

    def assign_prop_to_mesh(self, mesh, name, tensor):
        for j, val in enumerate(tensor):
            if len(tensor) > 1:
                mesh.point_arrays['%s_%d' % (name, j)] = \
                    val.squeeze().flatten('F')
            else:
                mesh.point_arrays['%s' % (name)] = \
                    val.squeeze().flatten('F')

    def save(self, mesh):
        files = self.directory.glob('%s*.%s' % (self.filename, self.extension))
        latest_file = "%s_%05d" % (self.filename, -1)
        for file in files:
            latest_file = file.stem
        file_number = latest_file.split('_')[-1]
        file_number = int(file_number)
        file_name = "%s_%05d.%s" % (
            self.filename, file_number + 1, self.extension)
        save_path = self.directory/file_name
        mesh.save(save_path)

    def save_grid(self, grid):
        files = self.directory.glob('%s*.npz' % self.filename)
        latest_file = "%s_%05d" % (self.filename, -1)
        for file in files:
            latest_file = file.stem
        file_number = latest_file.split('_')[-1]
        file_number = int(file_number)
        file_name = "%s_%05d.%s" % (
            self.filename, file_number + 1, 'npz')
        save_path = self.directory/file_name
        np.savez(save_path, **grid)

