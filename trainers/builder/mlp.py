from models import MLP
from .base import GraphBaseTrainer


class MLPTrainer(GraphBaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'], 
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'], 
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'], 
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])
        
    def build_model(self, args, **kwargs):
        model = MLP(
            input_size=args['input_size'],
            hidden_size=args['hidden_size'],
            output_size=args['output_size'],
            pos_dim=args['pos_dim'],
            num_layers=args['num_layers'],
            act=args['act'],
            batch_norm=args['batch_norm']
            )

        return model
