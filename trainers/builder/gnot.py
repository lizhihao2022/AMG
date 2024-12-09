from models import GNOT
from .base import GraphBaseTrainer


class GNOTTrainer(GraphBaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'], 
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'], 
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'], 
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])
        
    def build_model(self, args, **kwargs):
        model = GNOT(
            trunk_size=args['trunk_size'],
            branch_size=args['branch_size'],
            space_dim=args['space_dim'],
            output_size=args['output_size'],
            n_layers=args['n_layers'],
            n_hidden=args['n_hidden'],
            n_head=args['n_head'],
            n_inner=args['n_inner'],
            mlp_layers=args['mlp_layers'],
            attn_type=args['attn_type'],
            act=args['act'],
            ffn_dropout=args['ffn_dropout'],
            attn_dropout=args['attn_dropout'],
            horiz_fourier_dim=args['horiz_fourier_dim'],
            )

        return model
