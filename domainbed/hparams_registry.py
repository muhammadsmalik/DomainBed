# hparams_registry.py

import numpy as np
from domainbed.lib import misc

def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter."""
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet50_augmix', True, lambda r: True)
    _hparam('dinov2', False, lambda r: False)
    _hparam('vit', False, lambda r: False)
    _hparam('vit_attn_tune', False, lambda r: False)
    _hparam('freeze_bn', False, lambda r: False)
    _hparam('lars', False, lambda r: False)
    _hparam('linear_steps', 500, lambda r: 500)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('vit_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions.
    if algorithm == 'SIMPLE':
        _hparam('lr', 1e-4, lambda r: 10**r.uniform(-5, -3))
        _hparam('weight_decay', 0., lambda r: 0.)
        # Define any other algorithm-specific hyperparameters here.

    # Dataset-and-algorithm-specific hparam definitions.
    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif algorithm == 'RDM':
        if dataset in ['DomainNet', 'TerraIncognita']:
            _hparam('batch_size', 40, lambda r: int(r.uniform(30, 60)))
        else:
            _hparam('batch_size', 88, lambda r: int(r.uniform(70, 100)))
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))

    # Define 'weight_decay' if not already defined.
    if 'weight_decay' not in hparams:
        if dataset in SMALL_IMAGES:
            _hparam('weight_decay', 0., lambda r: 0.)
        else:
            _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    return hparams

def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}

def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
