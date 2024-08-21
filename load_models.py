import os
import sys
import torch
import torch.nn as nn
import utils

from earlyexit.earlyexit_model import EarlyExitModel

from models import cifar10 as cifar10_models
from models import cifar100 as cifar100_models
from models import waymo as waymo_models
from models import urban as urban_models
from models import video as video_models
from summary_graph import get_exits_def

import transformers
from models.deebert.src.modeling_deebert import DeeBertForSequenceClassification
from models.deebert.src.modeling_deedistilbert import DeeDistilBertForSequenceClassification
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DistilBertConfig,
    DistilBertTokenizer,
    # DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers.trainer_utils import is_main_process

# set transformers verbosity level
transformers.logging.set_verbosity_error()


def create_cifar10_model(arch, pretrained):
    try:
        model = cifar10_models.__dict__[arch]()
    except KeyError:
        raise ValueError(
            "Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def create_cifar100_model(arch, pretrained):
    try:
        model = cifar100_models.__dict__[arch]()
    except KeyError:
        raise ValueError(
            "Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def create_waymo_model(arch, pretrained):
    try:
        model = waymo_models.__dict__[arch](pretrained=pretrained)
    except KeyError:
        raise ValueError(
            "Model {} is not supported for dataset Waymo".format(arch))
    return model


def create_urban_model(arch, pretrained):
    try:
        model = urban_models.__dict__[arch](pretrained=pretrained)
        if 'resnet18_urban' in arch:
            model.fc = nn.Sequential(
                        nn.Linear(in_features=512, out_features=64, bias=True),
                        nn.Linear(in_features=64, out_features=3, bias=True)
                    )
        else:
            model.fc = nn.Sequential(
                        nn.Linear(in_features=2048, out_features=64, bias=True),
                        nn.Linear(in_features=64, out_features=3, bias=True)
                    )
    except KeyError:
        raise ValueError(
            "Model {} is not supported for dataset Urban".format(arch))
    return model

def create_video_model(arch, pretrained):
    try:
        model = video_models.__dict__[arch.split("_")[0]](pretrained=pretrained)
        if 'resnet18' in arch:
            model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        elif 'resnet50' in arch:
            model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    except KeyError:
        raise ValueError("Model {} is not supported".format(arch))
    return model


def create_deebert_model(model_type, dataset, weight_dir):
    """
    Args:    
        model_type (str): model architecture name
        model_size (str): base or large
        dataset (str): dataset name
        weight_dir (str): directory where the weights are stored
    """
    model_classes = {
        "bert-base-uncased": (BertConfig, DeeBertForSequenceClassification, BertTokenizer),
        "bert-large-uncased": (BertConfig, DeeBertForSequenceClassification, BertTokenizer),
        "distilbert-base-uncased": (DistilBertConfig, DeeDistilBertForSequenceClassification, DistilBertTokenizer),
        # TODO(ruipan): implement bp-based deeroberta?
    }
    config_class, model_class, tokenizer_class = model_classes[model_type]

    # set seed
    utils.set_seeds()

    weight_dir = os.path.join(
        weight_dir,
        model_type,
        dataset.upper(),
        "two_stage"
    )

    # prepare datasets
    dataset = dataset.lower()
    if dataset not in processors:
        raise ValueError(f"Task not found: {dataset}")
    processor = processors[dataset]()
    output_mode = output_modes[dataset]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # load pretrained model and tokenizer
    config = config_class.from_pretrained(
        weight_dir,
        num_labels=num_labels,
        finetuning_task=dataset,
        cache_dir=None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        weight_dir,
        do_lower_case=True,
        cache_dir=None,
    )
    model = model_class.from_pretrained(
        weight_dir,
        from_tf=bool(".ckpt" in weight_dir),
        config=config,
        cache_dir=None,  # can specify a str of the location to store the pre-trained models downloaded from huggingface.co
    )
    # in case the weights are not properly loaded, manually match the module names
    # (e.g., vanilla model loading from ee-enabled model's weights)
    model_state_dict = model.state_dict()
    expected_keys = list(model_state_dict.keys())
    loaded_state_dict = torch.load(
        f"{weight_dir}/pytorch_model.bin")  # , map_location="cpu"
    loaded_keys = list(loaded_state_dict.keys())
    keys_not_properly_loaded = set(expected_keys) - set(loaded_keys)
    # print(f"expected_keys {expected_keys}")
    # print(f"loaded_keys {loaded_keys}")
    # print(f"keys_not_properly_loaded {keys_not_properly_loaded}")
    # for key in keys_not_properly_loaded:
    #     if "branched_module" in key:  # ee-enabled model loading from vanilla model
    #         expected_key_wo_bp = key.replace("branched_module.", "")
    #     else:  # vanilla model loading from ee-enabled model
    #         # XXX(ruipan): replace all dots in str with ".branched_module.",
    #         # and search every one, but for now, hardcode
    #         expected_key_wo_bp = utils.nth_repl(
    #             key, ".", ".branched_module.", 4)
    #     assert loaded_state_dict[expected_key_wo_bp].shape == model_state_dict[key].shape, \
    #         f"Weight shape mismatch! loaded: {loaded_state_dict[expected_key_wo_bp].shape}, expected: {model_state_dict[key].shape}"
    #     model.state_dict()[key].data.copy_(
    #         loaded_state_dict[expected_key_wo_bp])
    #     print(f"Successfully copied weights into module {key}")

    return model, tokenizer, config


def create_model(pretrained, dataset, arch, weight_dir):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained (boolean): True is you wish to load a pretrained model.
            Some models do not have a pretrained version.
        dataset (string): dataset name (only 'imagenet' and 'cifar10' are supported)
        arch (string): architecture name
        weight_dir (str): directory where the weights are stored
        ids (list): list of ramp ids 

    Returns:
        model (torch.nn.Module): model
        tokenizer (transformer.tokenizer): tokenizer for language models. 
            Returns None for other workloads.
        bert_config (transformers.{BertConfig,RobertaConfig,DistilBertConfig}): 
            config required for initializing bert ramps
    """
    dataset = dataset.lower()
    model, tokenizer = None, None  # tokenizer is for NLP workloads
    bert_config = None
    cadene = False
    try:
        if dataset == 'cifar10':
            model = create_cifar10_model(arch, pretrained)
        elif dataset == 'cifar100':
            model = create_cifar100_model(arch, pretrained)
        elif dataset == 'waymo':
            model = create_waymo_model(arch, pretrained)
        elif dataset == 'urban':
            model = create_urban_model(arch, pretrained)
        elif dataset == 'video':
            model = create_video_model(arch, pretrained)
        elif dataset in ["mnli", "mrpc", "qnli", "qqp", "rte", "sst-2", "wnli"]:
            model, tokenizer, bert_config = create_deebert_model(
                arch, dataset, weight_dir)
    except ValueError:
        raise ValueError(
            'Could not recognize dataset {} and arch {} pair'.format(dataset, arch))

    model.arch = arch
    model.dataset = dataset
    return model, tokenizer, bert_config


def load_model(dataset, arch, weight_dir, num_classes=2, pretrained=False, earlyexit=True):
    """Load model.

    Args:
        dataset (str): dataset name
        arch (str): vanilla architecture name
        weight_dir (str): directory where the weights are stored
        pretrained (bool): True if you wish to load a pretrained model.
            Some models do not have a pretrained version.
        earlyexit (bool): True if 

    Returns:
        model (torch.nn.Module): EE model, but with vanilla weights and no ramps
        tokenizer (transformer.tokenizer): tokenizer for language models. 
            Returns None for other workloads.
        all_exit_def (list of (module_name, torch.nn.Module)): list of ramps with weights loaded
    """
    vanilla_model, tokenizer, bert_config = create_model(
        pretrained, dataset, arch, weight_dir)

    if earlyexit:
        if pretrained:
            # NOTE arch is the vanilla model arch
            weight_path = os.path.join(weight_dir, arch+"_earlyexit")
        else:
            weight_path = None
        if arch in utils.all_nlp_models:
            weight_path = os.path.join(
                weight_path[:weight_path.rfind("_")],
                dataset.upper(),
                "two_stage",
                "pytorch_model.bin",
            )

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ## load model and ramp weights
    # self.bert_config = bert_config
    # NOTE: the profile_path is the vanilla model's profile path, not ee model's
    profile_path = os.path.join(
        # os.getcwd(), f"profile_pickles/{arch}_profile.pickle")
        f"./profile_pickles_bs/{arch.split('_')[0]}_profile.pickle")
    if bert_config is None:
        module_name_prefix = None
    else:
        module_name_prefix_dict = {
            "bert": "bert.encoder.layer",
            "distilbert": "distilbert.transformer.layer"
        }
        module_name_prefix = module_name_prefix_dict[arch[:arch.find(
            '-')]]
        
    all_exit_def = get_exits_def(
        vanilla_model, arch, [-1], profile_path, num_classes, bert_config, module_name_prefix, dataset)

    load_weight(vanilla_model, arch, weight_path, all_exit_def, dataset)

    model = EarlyExitModel(vanilla_model) # wrap the vanilla model with EarlyExitModel
    model.to(device)
    
    return model, tokenizer, all_exit_def

def load_weight(model, arch, weight_dir_path, all_exit_def, dataset=None):
    """Load weight from weight_dir_path into model and all_exit_def
    
    Args:
        model (torch.nn.Module): model
        arch (str): vanilla architecture name
        weight_dir_path (str): directory where the weights are stored
        all_exit_def (list of (module_name, torch.nn.Module)): 
            list of ramps with weights not loaded yet
        dataset (str): dataset name
    
    Returns:

    """

    if weight_dir_path is not None:
        checkpoint = torch.load(weight_dir_path)
        model_dict = model.state_dict()
        
        expected_keys = list(model_dict.keys())
        checkpoint_keys = list(checkpoint.keys())
        ramp_name_state_map = {}

        for ramp_name, ramp_def in all_exit_def:
            ramp_name_state_map[ramp_name] = ramp_def.state_dict()
    
        if arch in utils.all_supported_models or dataset == 'video': 
            for checkpoint_key in checkpoint_keys:
                if "branch_net" in checkpoint_key:
                    l = checkpoint_key.split(".")
                    idx = l.index("branch_net")
                    ramp_name = ".".join(l[:idx])
                    ramp_state_key = ".".join(l[idx+1:])
                    ramp_name_state_map[ramp_name][ramp_state_key] = checkpoint[checkpoint_key]
                else:
                    if checkpoint_key in expected_keys:
                        key = checkpoint_key
                    elif "branched_module" in checkpoint_key:  # vanilla loading from EE checkpoint
                        key = checkpoint_key.replace("branched_module.", "")
                    assert checkpoint[checkpoint_key].shape == model_dict[key].shape, \
                        f"Weight shape mismatch! loaded: {checkpoint[checkpoint_key].shape}, expected: {model_dict[key].shape}"
                    model_dict[key].data.copy_(checkpoint[checkpoint_key])
                
        else:
            raise NotImplementedError
        
        
        for name, param in dict(model.named_parameters()).items():
            param.requires_grad = False

        for ramp_name, ramp_def in all_exit_def:
            ramp_def.load_state_dict(ramp_name_state_map[ramp_name])

    return model, all_exit_def


