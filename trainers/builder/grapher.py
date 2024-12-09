from models import Grapher
from .base import GraphBaseTrainer


class GrapherTrainer(GraphBaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'],
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'],
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'],
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])
    
    def build_model(self, args):
        model = Grapher(
            feature_width=args['feature_width'],
            num_layers=args['num_layers'],
            pos_dim=args['pos_dim'],
            input_features=args['input_features'],
            output_features=args['output_features'],
            batch_norm=args['batch_norm'],
            act=args['act'],
            global_ratio=args['global_ratio'],
            global_k=args['global_k'],
            global_cos=args['global_cos'],
            global_pos=args['global_pos'],
            local_nodes=args['local_nodes'],
            local_ratio=args['local_ratio'],
            local_k=args['local_k'],
            local_cos=args['local_cos'],
            local_pos=args['local_pos'],
            num_phys=args['num_phys'],
            num_heads=args['num_heads'],)
        
        return model
