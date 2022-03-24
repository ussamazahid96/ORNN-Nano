import os 
import platform

import torch
import argparse
from trainer import trainer


parser = argparse.ArgumentParser(description="Pytorch QORNN Training Project")


def none_or_str(value):
    if value == "None":
        return None 
    return value

def none_or_int(value):
    if value == "None":
        return None
    return int(value)

parser.add_argument("--dataset", default="AudioMNIST")
parser.add_argument("--datadir", default="./dataset", type=str)
parser.add_argument("--experiments", default="./experiments")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--log_freq", type=int, default=50)

parser.add_argument("--resume", type=none_or_str)
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--export", action="store_true")
parser.add_argument("--note", type=none_or_str)

parser.add_argument("--gpus", default='0', type=none_or_str)
parser.add_argument("--model", default="QORNN", type=none_or_str)
parser.add_argument("--num_workers", default=4, type=int)

parser.add_argument("--milestones", default='60', type=none_or_str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lr_orth", default=1e-4, type=float)
parser.add_argument("--ib", default=32, type=int)
parser.add_argument("--wb", default=32, type=int)
parser.add_argument("--rb", default=32, type=int)
parser.add_argument("--ab", default=32, type=int)

parser.add_argument("--scheduler", default='MultiStepLR', type=none_or_str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--random_seed", default="12345", type=none_or_int)


torch.set_printoptions(precision=10)



class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such Attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such Attribute: " + name)


if __name__=="__main__":
    args = parser.parse_args()
    # creating directories
    path_args = ["experiments", "resume"]
    for path_arg in path_args:
        path = getattr(args, path_arg)
        if path is not None and not os.path.isabs(path):
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            setattr(args, path_arg, abs_path)
    os.makedirs(args.experiments, exist_ok=True)

    # converting all arguments to dictionary
    if args.evaluate or args.export:
        args.dry_run = True 
    if not torch.cuda.is_available() or args.export:
        args.gpus = None
    if platform.system() == "Darwin":
        args.num_workers = 0 

    config = objdict(args.__dict__)
    trainer = trainer(config)
    
    if args.evaluate:
        trainer.eval_model()
    elif args.export:
        trainer.export()
    else:
        trainer.train_model()





