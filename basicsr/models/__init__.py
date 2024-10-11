import importlib
import os
from basicsr.utils import scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = os.path.dirname(os.path.abspath(__file__))
model_filenames = [
    os.path.splitext(os.path.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
]
# import all the model modules
_model_modules = [
    importlib.import_module(f'basicsr.models.{file_name}')
    for file_name in model_filenames
]


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt)

    return model
