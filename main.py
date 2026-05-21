import os
import torch
import random
import argparse
import numpy as np

from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer


def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    if cfg.model_dir is None:
        cfg.model_dir = cfg.output_dir
    setup_logger(cfg.output_dir, cfg.dataset)
    
    print("** Config **")
    print(cfg)
    print("************")
    
    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Cap GPU memory usage at 16GB (RTX 5070 Ti has 16GB total).
    if torch.cuda.is_available():
        gpu_id = cfg.gpu if cfg.gpu is not None else 0
        total_mib = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 2)
        cap_mib = 16 * 1024  # 16 GB hard cap
        frac = min(0.97, cap_mib / total_mib)
        torch.cuda.set_per_process_memory_fraction(frac, gpu_id)
        print(f"GPU memory capped to fraction {frac:.3f} (~{frac * total_mib / 1024:.1f} GiB) on cuda:{gpu_id}")

    trainer = Trainer(cfg)
    
    if cfg.zero_shot:
        trainer.test_zero_shot()
        return

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, help="data config file")
    parser.add_argument("--model", "-m", type=str, help="model config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
