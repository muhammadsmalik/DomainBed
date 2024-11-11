# algorithms.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import clip
import pretrainedmodels  # Import pretrainedmodels library
from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, ParamDict

ALGORITHMS = [
    'ERM',
    'Fish',
    'SIMPLE',  # Add your algorithm here
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

# Define Adapter
class Adapter(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Adapter, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim

    def forward(self, x):
        return self.fc(x)

# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=512):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embedding = self.model(x)
        return embedding

# Matching Network with Attention
class MatchingNetwork(nn.Module):
    def __init__(self, num_models, embedding_dim):
        super(MatchingNetwork, self).__init__()
        self.model_embeddings = nn.Embedding(num_models, embedding_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, sample_embedding):
        model_embeds = self.model_embeddings.weight  # (num_models, embedding_dim)
        attn_scores = torch.matmul(sample_embedding, model_embeds.t())  # (batch_size, num_models)
        attn_scores = attn_scores / self.temperature
        matching_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, num_models)
        return matching_weights

# Helper functions
def load_pretrained_model(model_name):
    if model_name in [
        'alexnet', 'densenet121', 'densenet169', 'densenet201',
        'resnet18', 'resnet34', 'resnet50', 'squeezenet1_0', 'squeezenet1_1'
    ]:
        model = getattr(models, model_name)(pretrained=True)
    elif model_name in ['dpn68', 'seresnet50', 'vit_base_patch16_224', 'vit_large_patch16_224', 'vit_giant_patch14_224_clip_laion2b']:
        model = timm.create_model(model_name, pretrained=True)
    elif model_name == 'nasnetamobile':
        model = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet')
    elif model_name in ['ViT-B/16', 'ViT-B/32']:
        model, _ = clip.load(model_name, device='cpu')  # Load on CPU first
        model = model.float()  # Ensure it's in float32 initially
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def modify_model(model, model_name):
    if model_name.startswith('resnet'):
        num_features = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name.startswith('densenet'):
        num_features = model.classifier.in_features
        model.classifier = nn.Identity()
    elif model_name == 'alexnet':
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        num_features = 4096
    elif model_name.startswith('squeezenet'):
        model.classifier = nn.Identity()
        num_features = 512  # Correct number after adaptive pooling
    elif model_name in ['dpn68', 'seresnet50']:
        model.reset_classifier(0)
        num_features = model.num_features
    elif model_name == 'nasnetamobile':
        num_features = model.last_linear.in_features
        model.last_linear = nn.Identity()
    elif model_name.startswith('vit'):
        num_features = getattr(model, 'num_features', None) or getattr(model, 'embed_dim', None)
        if num_features is None:
            raise AttributeError(f"{model_name} does not have 'num_features' or 'embed_dim' attributes.")
    elif model_name.startswith('ViT-'):  # CLIP models
        model = model.visual
        num_features = model.output_dim
        model.eval()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.num_features = num_features
    return model, num_features

def get_model_features(model, images, model_name):
    with torch.no_grad():
        if model_name.startswith('ViT-'):  # CLIP models
            # Convert both model and images to half precision
            model = model.half()
            features = model(images.half())
            features = features.float()  # Convert back to float32 for consistency
        elif model_name.startswith('squeezenet'):
            features = model.features(images)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        elif hasattr(model, 'forward_features'):
            features = model.forward_features(images)
        else:
            features = model(images)

        if features.dim() > 2:
            features = features.mean(dim=[2, 3]) if features.dim() == 4 else features[:, 0]

    return features.float()

class SIMPLE(Algorithm):
    """
    Implementation of the SIMPLE algorithm within DomainBed.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SIMPLE, self).__init__(input_shape, num_classes, num_domains, hparams)

        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(embedding_dim=512).to(self.device)

        # Define model names
        self.model_names = [
            'alexnet',
            'densenet121',
            'densenet169',
            'densenet201',
            'dpn68',
            'nasnetamobile',
            'resnet18',
            'resnet34',
            'resnet50',
            'seresnet50',
            'squeezenet1_0',
            'squeezenet1_1',
            'vit_base_patch16_224',
            'vit_large_patch16_224',
            'vit_giant_patch14_224_clip_laion2b',
            'ViT-B/16',  # CLIP model (uncomment if you have resources)
            'ViT-B/32',  # CLIP model (uncomment if you have resources)
        ]

        # Load and modify pretrained models
        self.pretrained_models = {}
        self.adapters = {}
        for model_name in self.model_names:
            print(f"Loading and modifying model {model_name}")
            model = load_pretrained_model(model_name)
            model, num_features = modify_model(model, model_name)
            model.eval().to(self.device)
            self.pretrained_models[model_name] = model

            # Create adapter
            adapter = Adapter(input_dim=num_features, num_classes=num_classes)
            adapter.to(self.device)
            self.adapters[model_name] = adapter

        # Initialize matching network
        num_models = len(self.model_names)
        embedding_dim = 512
        self.matching_network = MatchingNetwork(num_models=num_models, embedding_dim=embedding_dim).to(self.device)

        # Define optimizer and loss function
        params = list(self.matching_network.parameters())
        for adapter in self.adapters.values():
            params += list(adapter.parameters())

        self.optimizer = torch.optim.Adam(params, lr=hparams["lr"], weight_decay=hparams['weight_decay'])
        self.criterion = nn.CrossEntropyLoss()

    def update(self, minibatches, unlabeled=None):
        # Concatenate all minibatches
        all_x = torch.cat([x for x, y in minibatches]).to(self.device)
        all_y = torch.cat([y for x, y in minibatches]).to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Get sample embeddings
        with torch.no_grad():
            sample_embeddings = self.feature_extractor(all_x)

        # Get matching weights
        matching_weights = self.matching_network(sample_embeddings)

        # Get model outputs
        model_outputs = []
        for model_name in self.model_names:
            model = self.pretrained_models[model_name]
            adapter = self.adapters[model_name]
            features = get_model_features(model, all_x, model_name)
            features = features.to(self.device)
            logits = adapter(features)
            model_outputs.append(logits.unsqueeze(2))

        # Aggregate outputs
        model_outputs = torch.cat(model_outputs, dim=2)
        matching_weights_expanded = matching_weights.unsqueeze(1)
        aggregated_outputs = torch.sum(model_outputs * matching_weights_expanded, dim=2)

        # Compute loss and update
        loss = self.criterion(aggregated_outputs, all_y)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            sample_embeddings = self.feature_extractor(x)
            matching_weights = self.matching_network(sample_embeddings)
            model_outputs = []
            for model_name in self.model_names:
                model = self.pretrained_models[model_name]
                adapter = self.adapters[model_name]
                features = get_model_features(model, x, model_name)
                features = features.to(self.device)
                logits = adapter(features)
                model_outputs.append(logits.unsqueeze(2))
            model_outputs = torch.cat(model_outputs, dim=2)
            matching_weights_expanded = matching_weights.unsqueeze(1)
            aggregated_outputs = torch.sum(model_outputs * matching_weights_expanded, dim=2)
        return aggregated_outputs
