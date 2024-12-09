from .unet import UNetTrainer
from .grapher import GrapherTrainer
from .gnot import GNOTTrainer
from .mlp import MLPTrainer


TRAINER_DICT = {
    'MLP': MLPTrainer,
    'UNet': UNetTrainer,
    'GNOT': GNOTTrainer,
    'Grapher': GrapherTrainer,

    'mlp': MLPTrainer,
    'unet': UNetTrainer,
    'gnot': GNOTTrainer,
    'grapher': GrapherTrainer,
}
