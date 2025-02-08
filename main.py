import argparse
import traceback
import shutil
import logging
import yaml
import sys
from basicsr.utils import get_time_str
from basicsr.models import create_model
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import numpy as np
import torch.utils.tensorboard as tb

# from runners.diffusion import Diffusion
from guided_diffusion.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Use simplified DDNM, without SVD",
    )    
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--path_y",
        type=str,
        default='/eva_data1/shlu2240/Dataset/celeba_hq_02_4_10/test',
        # required=True,
        help="Path of the test dataset.",
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )    
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--deg_scale", type=float, default=0., help="deg_scale"
    )    
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--add_noise",
        action="store_true"
    )
    
    # kernel_estimator argument
    parser.add_argument(
        "--kernel_estimator",
        action="store_true"
    )
    parser.add_argument(
        "--Learning_degradation",
        action="store_true"
    )
    parser.add_argument(
        "--save_img",
        action="store_true"
    )
    parser.add_argument(
        "--sds",
        action="store_true"
    )
    parser.add_argument("--n_feats", type=int, default=64, help="conv channel")
    parser.add_argument("--kernel_size", type=int, default=21, help="predict kernel size")
    parser.add_argument("--kernel_model_path", type=str, default="/eva_data1/shlu2240/DDNM/exp/logs/kernel_estimator/estimator_x4.pt")
    parser.add_argument("--N", type=int, help="low/high path filter scale")

    parser.add_argument(
        '--IRopt', type=str, help='Path to option YAML file.')
    parser.add_argument(
        '--IRApopt', type=str, default="", help='Path to option YAML file.')
    parser.add_argument(
        '--dataset', type=str, help='testing dataset, imagenet or celeba or benchmark')
    parser.add_argument(
        '--mode', type=str, help="inference mode: explicit_gt | explicit | implicit | combine")
    parser.add_argument(
        "--perturb_y",
        action="store_true"
    )
    parser.add_argument(
        '--perturb_A', type=str, help='A use for perturb y, implicit or explicit or explicit_gt')
    parser.add_argument(
        '--perturb_Ap', type=str, help='Ap use for perturb y, implicit or explicit or explicit_gt')
    parser.add_argument(
        '--DDNM_A', type=str, help='A use for DDNM, implicit or explicit or explicit_gt')
    parser.add_argument(
        '--DDNM_Ap', type=str, help='Ap use for DDNM, implicit or explicit or explicit_gt')
    parser.add_argument(
        '--posterior_formula', type=str, default='DDIM', help='Ap use for DDNM, implicit or explicit or explicit_gt')
    parser.add_argument(
        '--input_mode', type=str, default='LQ', help='input LQ, direct LQ or AHQ')
    parser.add_argument(
        "--sds_optimize_alpha",
        action="store_true"
    )
    parser.add_argument(
        "--save_kernel",
        action="store_true"
    )
    parser.add_argument(
        "--sample_x0t",
        action="store_true"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0
    )
    parser.add_argument(
        "--total_step", type=int, default=100
    )
    parser.add_argument(
        "--check_consistency",
        action="store_true"
    )
    parser.add_argument(
        "--check_dist",
        action="store_true"
    )
    parser.add_argument(
        '--consistency_kernel', type=str, default="")
    parser.add_argument(
        "--sample_number", type=int, default=-1
    )
    parser.add_argument(
        '--diffusion_ckpt', type=str, default="/eva_data4/shlu2240/checkpoint/diffusion/256x256_diffusion_uncond.pt")    

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config['time_travel']['T_sampling'] = args.total_step
    new_config = dict2namespace(config)
    

    os.makedirs(os.path.join(args.exp, "result"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "result", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        response = input(
            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
        )
        if response.upper() == "Y":
            overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        

            
    log_file = os.path.join(f'{args.image_folder}', "log.txt")
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    # formatter = logging.Formatter(
    #     "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    # )
    # handler1.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w',
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S',
        )
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)



    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_options(args, is_train=False):
    with open(args.IRopt, "r") as config:
        opt = yaml.safe_load(config)
    opt['is_train'] = is_train
    opt['name'] = f"{opt['name']}_{get_time_str()}"

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = os.path.expanduser(val)
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_states'] = os.path.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = os.path.join(experiments_root,
                                                'visualization')
        # os.makedirs(opt['path']['experiments_root'], exist_ok=True)
        # os.makedirs(opt['path']['models'], exist_ok=True)
        # os.makedirs(opt['path']['training_states'], exist_ok=True)
        # os.makedirs(opt['path']['visualization'], exist_ok=True)
        # os.makedirs(opt['path']['log'], exist_ok=True)
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = os.path.join(results_root, 'visualization')

        # os.makedirs(opt['path']['results_root'], exist_ok=True)
        # os.makedirs(opt['path']['log'], exist_ok=True)
        # os.makedirs(opt['path']['visualization'], exist_ok=True)
    return opt

def init_loggers(args):
    log_file = os.path.join(f'{args.image_folder}', "log.txt")
    logger = logging.getLogger(name='DDNM')
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w',
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S',
        )
    return logger

def main():
    args, config = parse_args_and_config()
    txt_logger = init_loggers(args)
    
    # Convert the dictionary to JSON format
    argparse_dict = vars(args)
    with open(os.path.join(args.image_folder, 'args.json'), 'w+') as f:
        json.dump(argparse_dict, f, indent=4) 

    
    if args.sds:
        runner = Diffusion(args, config, txt_logger=txt_logger)
        runner.sds()      
          
    elif args.Learning_degradation:
        IRopt = parse_options(args)
        IRmodel = create_model(IRopt)
        IRmodel2 = None
        
        if args.IRApopt != "":
            with open(args.IRApopt, "r") as config2:
                opt2 = yaml.safe_load(config2)
                opt2['is_train'] = False
                IRmodel2 = create_model(opt2)

        try:
            runner = Diffusion(args, config, IRmodel=IRmodel, IRopt=IRopt, txt_logger=txt_logger, IRmodel2=IRmodel2)
            runner.sample(args.simplified)
        except Exception:
            logging.error(traceback.format_exc())
    else:
        try:
            runner = Diffusion(args, config, txt_logger=txt_logger)
            runner.sample(args.simplified)
        except Exception:
            logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
