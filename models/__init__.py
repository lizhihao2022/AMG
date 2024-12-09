from .unet import UNet1d
from .grapher import Grapher
from .gnot import GNOT
from .mlp import MLP


ALL_MODELS = {
    'UNet': UNet1d,
    'Grapher': Grapher,
    'GNOT': GNOT,   
    'MLP': MLP,
    
    'unet': UNet1d,
    'grapher': Grapher,
    'gnot': GNOT,
    'mlp': MLP,
}
