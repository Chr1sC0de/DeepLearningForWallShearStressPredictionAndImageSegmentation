from . import CallOn
from pathlib import Path
from typing import Iterable
import pyvista as pv


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

    def save_fiel(self, env):
        NotImplemented

# class SaveDictionary(SaveIntermediatePred):
#     pass
    #to be done

class SavePyvistaPoints(SaveIntermediate):
    extension = 'vtk'

    def __init__(
            self, *args, properties,
            points_key='points', on_end=False, **kwargs):
        super(SavePyvistaPoints, self).__init__(*args, on_end=on_end, **kwargs)
        assert isinstance(properties, Iterable), 'properties must be iterable'
        self.properties = properties
        self.points_key = points_key

    def method(self, env):
        self.construct_to_save(env)
        for mesh in self.meshes_to_save:
            self.save(mesh)

    def construct_to_save(self, env):
        batch_points = env.batch[self.points_key].detach().to('cpu').numpy()
        meshes = []
        for i, points in enumerate(batch_points):
            mesh = pv.StructuredGrid(
                points[0], points[1], points[2])
            for prop in self.properties:
                data_prop = env.batch[prop][i].detach().to('cpu').numpy()
                self.assign_prop_to_mesh(mesh, prop, data_prop)
            for prop, name in zip([env.y_true, env.y_pred], ['y_true', 'y_pred']):
                prop = prop.detach().to('cpu').numpy()[i]
                self.assign_prop_to_mesh(mesh, name, prop)
            meshes.append(mesh)
        self.meshes_to_save = meshes

    def assign_prop_to_mesh(self, mesh, name, tensor):
        for j, val in enumerate(tensor):
            if len(tensor) > 1:
                mesh.point_arrays['%s_%d' % (name, j)] = \
                    val.squeeze().flatten('F')
            else:
                mesh.point_arrays['%s' % (name)] = \
                    val.squeeze().flatten('F')

    def save(self, mesh):
        files = self.directory.glob('%s*' % self.filename)
        latest_file = "%s_%05d" % (self.filename, -1)
        for file in files:
            latest_file = file.stem
        file_number = latest_file.split('_')[-1]
        file_number = int(file_number)
        file_name = "%s_%05d.%s" % (
            self.filename, file_number + 1, self.extension)
        save_path = self.directory/file_name
        mesh.save(save_path)
