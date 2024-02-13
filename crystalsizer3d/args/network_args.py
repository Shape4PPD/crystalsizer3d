from argparse import ArgumentParser, Namespace
from typing import Union

from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.util.utils import str2bool

NETWORK_TYPES = ['densenet', 'fcnet', 'resnet', 'pyramidnet', 'vitnet', 'timm']


class NetworkArgs(BaseArgs):
    def __init__(
            self,
            base_net: str = None,
            hyperparameters: dict = None,
            **kwargs
    ):
        assert base_net in NETWORK_TYPES, f'Base network must be one of {NETWORK_TYPES}.'
        self.base_net = base_net
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = hyperparameters

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str = None):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group(
            'Network Args' + (f'_Prefix={prefix}' if prefix is not None else '')
        )
        if prefix is not None:
            prefix = prefix + '-'
        else:
            prefix = ''

        # Add network-specific options
        # if prefix == '':
        #     subparsers = parser.add_subparsers(title='Base network', dest=f'base_net',
        #                                        help='What type of network to use as the base.')
        #     subparsers.required = True
        # else:
        subparsers = parser.add_subparsers(title='Base network', dest=f'{prefix.replace("-", "_")}base_net',
                                           help='What type of network to use as the base.')
        subparsers.required = True
        NetworkArgs._add_fcnet_args(subparsers.add_parser('fcnet'), prefix)
        NetworkArgs._add_resnet_args(subparsers.add_parser('resnet'), prefix)
        NetworkArgs._add_densenet_args(subparsers.add_parser('densenet'), prefix)
        NetworkArgs._add_pyramidnet_args(subparsers.add_parser('pyramidnet'), prefix)
        NetworkArgs._add_vitnet_args(subparsers.add_parser('vitnet'), prefix)
        NetworkArgs._add_timm_args(subparsers.add_parser('timm'), prefix)

        return group

    @staticmethod
    def _add_fcnet_args(parser: Namespace, prefix: str = None):
        """
        Fully-connected network parameters.
        """
        parser.add_argument(f'--{prefix}layers-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='100,100',
                            help='Comma delimited list of layer sizes.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_resnet_args(parser, prefix: str = None):
        """
        ResNet network parameters.
        """
        parser.add_argument(f'--{prefix}n-init-channels', type=int, default=64,
                            help='Number of channels in the first input layer.')
        parser.add_argument(f'--{prefix}blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='3,4,6,3',
                            help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
        parser.add_argument(f'--{prefix}shortcut-type', type=str, choices=['id', 'conv'], default='id',
                            help='Shortcut operation to use when dimensions change.')
        parser.add_argument(f'--{prefix}use-bottlenecks', type=str2bool, default=False,
                            help='Use bottleneck type residual layers.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_densenet_args(parser, prefix: str = None):
        """
        DenseNet network parameters.
        """
        parser.add_argument(f'--{prefix}n-init-channels', type=int, default=16,
                            help='Number of channels in the first input layer.')
        parser.add_argument(f'--{prefix}growth-rate', type=int, default=8,
                            help='Growth rate for each layer in the blocks (k).')
        parser.add_argument(f'--{prefix}compression-factor', type=float, default=0.5,
                            help='Factor to reduce resolution by in transition layers (theta).')
        parser.add_argument(f'--{prefix}blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='6,6,6',
                            help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_pyramidnet_args(parser, prefix: str = None):
        """
        PyramidNet network parameters.
        """
        parser.add_argument(f'--{prefix}n-init-channels', type=int, default=16,
                            help='Number of channels in the first input layer.')
        parser.add_argument(f'--{prefix}blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='3,4,6,3',
                            help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
        parser.add_argument(f'--{prefix}alpha', type=int, default=420,
                            help='The widening factor which defines how quickly the pyramid expands at each layer.')
        parser.add_argument(f'--{prefix}shortcut-type', type=str, choices=['id', 'conv'], default='id',
                            help='Shortcut operation to use when dimensions change.')
        parser.add_argument(f'--{prefix}use-bottlenecks', type=str2bool, default=False,
                            help='Use bottleneck type residual layers.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_vitnet_args(parser: Namespace, prefix: str = None):
        """
        ViT network parameters.
        """
        parser.add_argument(f'--{prefix}model-name', type=str, default='google/vit-base-patch16-224-in21k',
                            help='Name of the model to use.')
        parser.add_argument(f'--{prefix}use-cls-token', type=str2bool, default=True,
                            help='Whether to use the CLS token only or the full image embedding.')
        parser.add_argument(f'--{prefix}classifier-hidden-layers', type=lambda s: [int(item) for item in s.split(',')],
                            default=[],
                            help='Comma delimited list of hidden layer sizes to use in the classifier.')
        parser.add_argument(f'--{prefix}vit-dropout-prob', type=float, default=0.,
                            help='Dropout probability to use in the transformer layers.')
        parser.add_argument(f'--{prefix}classifier-dropout-prob', type=float, default=0.,
                            help='Dropout probability to use before classifier layers.')

    @staticmethod
    def _add_timm_args(parser: Namespace, prefix: str = None):
        """
        Pretrained network parameters.
        """
        parser.add_argument(f'--{prefix}model-name', type=str, default='resnet51q',
                            help='Name of the model to use.')
        parser.add_argument(f'--{prefix}classifier-hidden-layers', type=lambda s: [int(item) for item in s.split(',')],
                            default=[],
                            help='Comma delimited list of hidden layer sizes to use in the classifier.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.,
                            help='Dropout probability to use in the pretrained network.')
        parser.add_argument(f'--{prefix}droppath-prob', type=float, default=0.,
                            help='Drop-path probability to use in the pretrained network.')
        parser.add_argument(f'--{prefix}classifier-dropout-prob', type=float, default=0.,
                            help='Dropout probability to use before classifier layers.')

    @staticmethod
    def extract_hyperparameter_args(args: Namespace) -> dict:
        """
        Create a NetworkParameters instance from command-line arguments.
        """
        if args.base_net == 'fcnet':
            hyperparameters = {
                'layers_config': args.layers_config,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'resnet':
            hyperparameters = {
                'n_init_channels': args.n_init_channels,
                'blocks_config': args.blocks_config,
                'shortcut_type': args.shortcut_type,
                'use_bottlenecks': args.use_bottlenecks,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'densenet':
            hyperparameters = {
                'n_init_channels': args.n_init_channels,
                'growth_rate': args.growth_rate,
                'compression_factor': args.compression_factor,
                'blocks_config': args.blocks_config,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'pyramidnet':
            hyperparameters = {
                'n_init_channels': args.n_init_channels,
                'blocks_config': args.blocks_config,
                'alpha': args.alpha,
                'shortcut_type': args.shortcut_type,
                'use_bottlenecks': args.use_bottlenecks,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'vitnet':
            hyperparameters = {
                'model_name': args.model_name,
                'use_cls_token': args.use_cls_token,
                'classifier_hidden_layers': args.classifier_hidden_layers,
                'vit_dropout_prob': args.vit_dropout_prob,
                'classifier_dropout_prob': args.classifier_dropout_prob,
            }

        elif args.base_net == 'timm':
            hyperparameters = {
                'model_name': args.model_name,
                'dropout_prob': args.dropout_prob,
                'droppath_prob': args.droppath_prob,
                'classifier_hidden_layers': args.classifier_hidden_layers,
                'classifier_dropout_prob': args.classifier_dropout_prob,
            }

        return hyperparameters

    @classmethod
    def from_args(cls, args: Union[Namespace, dict]) -> 'NetworkArgs':
        """
        Create a NetworkParameters instance from command-line arguments.
        """
        if isinstance(args, dict):
            args = Namespace(**args)
        if hasattr(args, 'hyperparameters'):
            hyperparameters = args.hyperparameters
        else:
            hyperparameters = NetworkArgs.extract_hyperparameter_args(args)
        return NetworkArgs(
            base_net=args.base_net,
            hyperparameters=hyperparameters,
        )
