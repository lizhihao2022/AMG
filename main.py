from config import parser
from utils import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config
from trainers.procedure import *


def main():
    args = parser.parse_args()
    args = vars(args)
    args = load_config(args)
    
    if args['log'] is True:
        saving_path, saving_name = set_up_logger(args["model_name"], args["dataset"], args['log_dir'])
    elif args.save_best is True or args.save_check_points is True:
        saving_path, saving_name = get_dir_path(args["model_name"], args["dataset"], args['log_dir'])
    else:
        saving_path, saving_name = None, None
    
    args['saving_path'] = saving_path
    args['saving_name'] = saving_name
    save_config(args, saving_path)
    set_device(args['cuda'], args['device'])
    set_seed(args['random_seed'])

    if args['dataset'] == 'NavierStokes':
        ns_procedure(args)
    elif args['dataset'] == 'CylinderFlow':
        cylinder_flow_procedure(args)
    elif args['dataset'] == 'AirfoilTime':
        airfoil_time_procedure(args)
    elif args['dataset'] == 'DeformingPlate':
        deforming_plate_procedure(args)
    elif args['dataset'] == 'Poisson':
        poisson_procedure(args)
    elif args['dataset'] == 'CarCFD':
        car_cfd_procedure(args)
    elif args['dataset'] == 'Climate':
        climate_procedure(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
