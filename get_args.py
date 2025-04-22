from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional
from torch import optim
from torch import nn
from models.graph_T5.classifier import GraphT5Classifier
import logging

def add_args_shared(parser: ArgumentParser):
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        help="wandb mode. For example `disabled` to disable wandb, which can be useful for debugging.",
    )
    parser.add_argument(
        "--kg",
        type=str,
        default="conceptnet",
        help="name of the knowledge graph",
    )
    parser.add_argument(
        "--dataset_construction",
        type=str,
        default="semantic",
        help="how the dataset is constructed. 'semantic' means that the dataset is constructed by selecting neighbors according to their semantic similarity. 'random' means that the dataset is constructed by sampling neighbors uniformly.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="radius of the subgraphs. e.g. 1 means that only one triplet is in the subgraph",
    )
    parser.add_argument(
        "--num_masked",
        type=int,
        default=0,
        help="size of masked subgraph. 0 means that only the relation to be predicted is masked. 1 means that neighboring concepts are masked as well. 2 means that additionally the next relations are masked. 3 means that additionally the next concepts are masked. etc.",
    )
    parser.add_argument(
        "--modelsize",
        type=str,
        default="t5-base",
        help="size of the model",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3",
        help="device",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="number of epochs",
    )
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=5,
        help='number of epochs without improvement before stopping training',
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--optimizer",
        type=str2optimizer,
        default="AdamW",
        help="optimizer",
    )
    parser.add_argument(
        "--criterion",
        type=str2criterion,
        default="CrossEntropyLoss",
        help="criterion, i.e. loss function",
    )
    parser.add_argument(
        "--logging_level",
        type=str2logging_level,
        default="INFO",
        help="logging level",
    )
    parser.add_argument(
        "--wandb_name_prefix",
        type=str,
        default="",
        help="prefix to run name in wandb",
    )

def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--params_to_train",
        type=str,
        default="all",
        help="which parameters to train. 'all' means all parameters. 'head' means only the parameters that are added on top of the pretrained model.",
    )
    parser.add_argument(
        "--graph_representation",
        type=str,
        default="gGLM",
        help="How the graph is represented. 'lGLM' means local graph language model. 'set' means that the graph is represented as a set of triplets (random order) and that the model is a sequence model. 'gGLM' means global GLM, i.e. the same as lGLM but the attention is not sparse and non-neighboring relations and concepts have a PE of the maximum distance. 'list' means that the graph is represented as a list of triplets (alphabetical oder) and that the model is a sequence model.",
    )
    parser.add_argument(
        "--reset_params",
        type=str2bool,
        default=False,
        help="whether to reset the parameters of the model before training. This removes pretrained weights.",
    )
    parser.add_argument(
        "--reload_data",
        type=str2bool,
        default=None,
        help="whether to reload the data in every training epoch. If None, then the default value is chosen depending on the value of graph_representation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps. Effective batch size is `train_batch_size * gradient_accumulation_steps`",
    )
    parser.add_argument(
        "--eos_usage",
        type=str,
        default="False",
        help="Only relevant when using GLM. eos stands for end-of-sequence token. Can be `False` for not using an eos token. When using an eos token, there are two ways to use it: `bidirectional` means that the eos token is connected to every other node in the graph, with a relative position of positive infinity (from node to eos) or negative infinity (from eos to node). `unidirectional` means that the eos token is connected to every node in the graph with a relative position of positive infinity (from node to eos), but not the other way around (i.e. no connection from eos to other node). This means, that nodes do not get messages from the eos token, which perceives locality when using the local GLM"
    )
    parser.add_argument(
        "--num_additional_buckets",
        type=int,
        default=None,
        help="number of additional buckets for relative position embedding. If None, then the default depending on the graph_representation is chosen."
    )
    parser.add_argument(
        "--init_additional_buckets_from",
        type=str2int,
        default=1e6,
        help="Specifies from which bucket of the parent model the additional buckets are initialized. init_additional_buckets_from gives the relative position, and the bucket is the one which corresponds to that relative position. If None, then the additional buckets are initialized randomly as determined by from_pretrained().",
    )

def load_args(parser: ArgumentParser):
    args = parser.parse_args()
    if args.reload_data is None:
        if args.graph_representation in ["set"]:
            args.reload_data = True
        elif args.graph_representation in ["lGLM", "gGLM", "list"]:
            args.reload_data = False
        else:
            raise ValueError(f"unknown graph_representation {args.graph_representation}")

    if args.num_additional_buckets is None:
        if args.graph_representation in ["set", "list","lGLM"]:
            args.num_additional_buckets = 0
        elif args.graph_representation in ["gGLM"]:
            args.num_additional_buckets = 1
        else:
            raise ValueError(f"unknown graph_representation {args.graph_representation}")
    if args.eos_usage != 'False' and args.graph_representation not in ['lGLM', 'gGLM']:
        raise ValueError(f"eos_usage can only be used with lGLM or gGLM, but not with {args.graph_representation}")
    return args

def str2bool(s:str)->bool:
    # input can also be bool
    if s in ['True', 'true', '1', True]:
        return True
    elif s in ['False', 'false', '0', False]:
        return False
    elif s in ['None', None]:
        return None
    else:   
        raise ValueError(f"unknown boolean value {s}")

def str2int(s:str)->Optional[int]:
    if s in ['None', None]:
        return None
    else:
        return int(float(s))

def str2optimizer(s:str)->optim.Optimizer:
    if s == "Adam":
        return optim.Adam
    elif s == "SGD":
        return optim.SGD
    elif s == "AdamW":
        return optim.AdamW
    else:
        raise ValueError(f"unknown optimizer {s}")

def str2criterion(s:str)->nn.Module:
    if s == "CrossEntropyLoss":
        return nn.CrossEntropyLoss
    else:
        raise ValueError(f"unknown criterion {s}")
    

def freeze_params(s:str, model:GraphT5Classifier) -> None:
    """
    :param s: string that specifies which parameters to train
    :param model: model
    """
    if s == "all":  # all
        pass
    elif s == "head":  # only head
        for param in model.t5model.parameters():
            param.requires_grad = False
    elif s == "PE":  # PE and head
        for param in model.t5model.parameters():  # set all encoder parameters to not trainable
            param.requires_grad = False
        for l in model.t5model.encoder.block:  # set PE to trainable
            if l.layer[0].SelfAttention.has_relative_attention_bias:
                logging.info('setting relative_attention_bias to trainable')
                l.layer[0].SelfAttention.relative_attention_bias.requires_grad = True
                assert l.layer[0].SelfAttention.relative_attention_bias.requires_grad
    else:
        raise ValueError(f"unknown parameters_to_train {s}")

def reset_params(model:GraphT5Classifier) -> None:
    for param in model.parameters():
        nn.init.normal_(param)
    for module in model.t5model.modules():
        model.t5model._init_weights(module)

def str2logging_level(s:str):
    if s == "CRITICAL":
        return logging.CRITICAL
    elif s == "WARNING":
        return logging.WARNING
    elif s == "INFO":
        return logging.INFO
    elif s == "DEBUG":
        return logging.DEBUG
    else:
        raise ValueError(f"unknown logging_level {s}")
