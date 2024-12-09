from models import UNet1d
from .base import GraphBaseTrainer


class UNetTrainer(GraphBaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'], 
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'], 
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'], 
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])
        
    def build_model(self, args, **kwargs):
        model = UNet1d(
            in_channels=args['in_channels'],
            out_channels=args['out_channels'],
            init_features=args['init_features'],
            pos_dim=args['pos_dim'],
            )

        return model
