import json
import time
from pathlib import Path
from typing import Optional

import torch

from crystalsizer3d import logger
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs
from crystalsizer3d.args.denoiser_args import DenoiserArgs
from crystalsizer3d.args.generator_args import GeneratorArgs
from crystalsizer3d.args.network_args import NetworkArgs
from crystalsizer3d.args.optimiser_args import OptimiserArgs
from crystalsizer3d.args.runtime_args import RuntimeArgs
from crystalsizer3d.args.transcoder_args import TranscoderArgs
from crystalsizer3d.nn.dataset import Dataset
from crystalsizer3d.util.utils import hash_data


class Checkpoint:
    def __init__(
            self,
            dataset: Dataset,
            dataset_args: DatasetTrainingArgs,
            network_args: NetworkArgs,
            generator_args: GeneratorArgs,
            denoiser_args: DenoiserArgs,
            transcoder_args: TranscoderArgs,
            optimiser_args: OptimiserArgs,
            runtime_args: RuntimeArgs,
            save_dir: Path
    ):
        self.dataset = dataset
        self.dataset_args = dataset_args
        self.network_args = network_args
        self.generator_args = generator_args
        self.denoiser_args = denoiser_args
        self.transcoder_args = transcoder_args
        self.optimiser_args = optimiser_args
        self.runtime_args = runtime_args
        self.save_dir = save_dir
        self._init_data()

    @property
    def id(self):
        return hash_data({
            'dataset_args': self.dataset_args.hash(),
            'network_args': self.network_args.hash(),
            'generator_args': self.generator_args.hash(),
            'denoiser_args': self.denoiser_args.hash(),
            'transcoder_args': self.transcoder_args.hash(),
            'optimiser_args': self.optimiser_args.hash(),
        })

    @property
    def path(self):
        return self.save_dir / f'{self.id}_{self.created}.json'

    def _init_data(self):
        """
        Initialise the disk-backed data.
        """
        self.created = time.strftime('%Y%m%d%H%M')
        self.epoch = 0
        self.step = 0
        self.examples_count = 0
        self.loss_train = 1e10
        self.loss_test = 1e10
        self.metrics_train = {}
        self.metrics_test = {}
        self.snapshots = {}
        self.old_args = {}

        # Load the data from disk if it exists
        resume_from = None
        if self.runtime_args.resume:
            if self.runtime_args.resume_from == 'latest':
                # Load the timestamps of previous runs from their filenames
                timestamps = []
                for f in self.save_dir.glob(f'{self.id}*.json'):
                    ts = int(f.name.split('_')[-1].split('.')[0])
                    timestamps.append(ts)

                if len(timestamps) == 0:
                    logger.warning(f'No checkpoints found in {self.save_dir}.')

                else:
                    # Load the most recent checkpoint
                    ts = max(timestamps)
                    resume_from = str(ts)

            else:
                resume_from = self.runtime_args.resume_from

            if resume_from is not None:
                if resume_from.isdigit():
                    load_path = self.save_dir / f'{self.id}_{resume_from}.json'
                else:
                    load_path = Path(resume_from)
                    assert load_path.suffix == '.json', f'Checkpoint {resume_from} is not a json file.'
                assert load_path.exists(), f'Checkpoint {load_path} does not exist.'
                with open(load_path, 'r') as f:
                    data = json.load(f)
                old_args = {}
                for k, v in data.items():
                    if hasattr(self, k) and k not in ['id', 'path']:
                        if k == 'snapshots':
                            # Convert snapshot dict keys to ints
                            for key, val in v.items():
                                self.snapshots[int(key)] = val
                        elif k == 'old_args':
                            self.old_args = v
                        elif k.endswith('_args'):
                            # Keep a log of any changed args
                            if k == 'dataset_args':
                                old_args_k = DatasetTrainingArgs.from_args(v)
                                new_args_k = self.dataset_args
                            elif k == 'network_args':
                                old_args_k = NetworkArgs.from_args(v)
                                new_args_k = self.network_args
                            elif k == 'generator_args':
                                old_args_k = GeneratorArgs.from_args(v)
                                new_args_k = self.generator_args
                            elif k == 'denoiser_args':
                                old_args_k = DenoiserArgs.from_args(v)
                                new_args_k = self.denoiser_args
                            elif k == 'transcoder_args':
                                old_args_k = TranscoderArgs.from_args(v)
                                new_args_k = self.transcoder_args
                            elif k == 'optimiser_args':
                                old_args_k = OptimiserArgs.from_args(v)
                                new_args_k = self.optimiser_args
                            if old_args_k.hash() != new_args_k.hash():
                                old_args[k] = old_args_k.to_dict()
                        else:
                            setattr(self, k, v)
                if len(old_args) > 0:
                    self.old_args[time.strftime('%Y%m%d%H%M%S')] = old_args
                logger.info(f'Loaded checkpoint data from {load_path}, created={self.created}')
                logger.info(f'Test loss = {self.loss_test:.4E}')
                for key, val in self.metrics_test.items():
                    logger.info(f'\t{key}: {val:.4E}')

        # Otherwise, create a new checkpoint
        if resume_from is None:
            logger.info(f'Creating new checkpoint at {self.path}.')
            self.save(create_snapshot=False)

    def save(
            self,
            create_snapshot: bool = True,
            net_p: Optional[torch.nn.Module] = None,
            optimiser_p: Optional[torch.optim.Optimizer] = None,
            net_g: Optional[torch.nn.Module] = None,
            optimiser_g: Optional[torch.optim.Optimizer] = None,
            net_d: Optional[torch.nn.Module] = None,
            optimiser_d: Optional[torch.optim.Optimizer] = None,
            net_dn: Optional[torch.nn.Module] = None,
            optimiser_dn: Optional[torch.optim.Optimizer] = None,
            net_t: Optional[torch.nn.Module] = None,
            optimiser_t: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Save the checkpoint to disk.
        """
        meta = {'created': self.created, 'id': self.id}
        for args_key in ['dataset_args', 'network_args', 'generator_args', 'denoiser_args',
                         'transcoder_args', 'optimiser_args', 'runtime_args']:
            args = getattr(self, args_key)
            args_dict = args.to_dict()
            args_dict['hash'] = args.hash()
            meta[args_key] = args_dict
        meta['old_args'] = self.old_args

        # Clean the data before saving
        self._clean()
        stats = {
            'epoch': self.epoch,
            'step': self.step,
            'examples_count': self.examples_count,
            'loss_train': self.loss_train,
            'loss_test': self.loss_test,
            'metrics_train': self.metrics_train,
            'metrics_test': self.metrics_test,
        }

        # Create a snapshot if required
        if create_snapshot:
            if self.dataset_args.train_predictor:
                assert net_p is not None, 'Must provide the predictor network to create a snapshot.'
                assert optimiser_p is not None, 'Must provide the predictor network optimiser to create a snapshot.'
            if self.dataset_args.train_generator:
                assert net_g is not None, 'Must provide the generator network to create a snapshot.'
                if not self.dataset_args.train_combined:
                    assert optimiser_g is not None, 'Must provide the generator network optimiser to create a snapshot.'
                if self.generator_args.use_discriminator:
                    assert net_d is not None, 'Must provide the discriminator network to create a snapshot.'
                    assert optimiser_d is not None, 'Must provide the discriminator network optimiser to create a snapshot.'
            if self.dataset_args.train_denoiser:
                assert net_dn is not None, 'Must provide the denoiser network to create a snapshot.'
                assert optimiser_dn is not None, 'Must provide the denoiser network optimiser to create a snapshot.'
            if self.transcoder_args.use_transcoder:
                assert net_t is not None, 'Must provide the transcoder network to create a snapshot.'

            # Create the snapshot entry for the json record
            snapshot = stats.copy()
            snapshot['created'] = time.strftime('%Y%m%d%H%M%S')
            snapshot_dir = Path(self.save_dir / self.id / 'snapshots')
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            state_path = snapshot_dir / f'{snapshot["step"]:09d}_{snapshot["created"]}.pth'
            snapshot['state_path'] = str(state_path.relative_to(self.save_dir))
            self.snapshots[self.step] = snapshot

            # Save the networks and optimisers state parameters
            data = {}
            if self.dataset_args.train_predictor:
                # If we're using a frozen pre-trained model then only save the classifier parameters
                if self.network_args.base_net in ['vitnet', 'timm'] and self.optimiser_args.freeze_pretrained:
                    net_p = net_p.classifier
                data['net_p_state_dict'] = net_p.state_dict()
                data['optimiser_p_state_dict'] = optimiser_p.state_dict()
            if self.dataset_args.train_generator:
                data['net_g_state_dict'] = net_g.state_dict()
                if not self.dataset_args.train_combined:
                    data['optimiser_g_state_dict'] = optimiser_g.state_dict()
                if self.generator_args.use_discriminator:
                    data['net_d_state_dict'] = net_d.state_dict()
                    data['optimiser_d_state_dict'] = optimiser_d.state_dict()
            if self.dataset_args.train_denoiser:
                data['net_dn_state_dict'] = net_dn.state_dict()
                data['optimiser_dn_state_dict'] = optimiser_dn.state_dict()
            if self.transcoder_args.use_transcoder:
                data['net_t_state_dict'] = net_t.state_dict()
                if optimiser_t is not None:
                    data['optimiser_t_state_dict'] = optimiser_t.state_dict()
            torch.save(data, state_path)
            logger.info(f'Saved model parameters to {state_path}.')

            # Remove the old snapshots if there are too many
            while len(self.snapshots) > self.runtime_args.max_checkpoints:
                step = min(list(self.snapshots.keys()))
                snapshot = self.snapshots[step]

                # Check the snapshot path corresponds to this id as it might be resumed from a different run
                if snapshot['state_path'].split('/')[0] == self.id:
                    state_path = self.save_dir / snapshot['state_path']
                    state_path.unlink()
                    logger.info(f'Removed snapshot {state_path}.')

                del self.snapshots[step]

        # Save the checkpoint to disk
        data = {**meta, **stats, 'snapshots': self.snapshots}
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f'Saved checkpoint data to {self.path}.')

    def _clean(self):
        """
        Fix the metric values to be standard floats rather than torch tensors.
        """
        for k, v in self.metrics_train.items():
            self.metrics_train[k] = float(v)
        for k, v in self.metrics_test.items():
            self.metrics_test[k] = float(v)

    def get_state_path(self, step: Optional[int] = None) -> Path:
        """
        Get the path to the state file for the current step.
        """
        if len(self.snapshots) == 0:
            raise RuntimeError('No snapshots found.')

        if step is None:
            # Use the latest snapshot
            step = max(list(self.snapshots.keys()))
            snapshot = self.snapshots[step]
        else:
            snapshot = self.snapshots[step]

        # If we're resuming from a checkpoint then look for the snapshot relative to it
        resume_from = self.runtime_args.resume_from
        if resume_from is not None and not resume_from.isdigit() and not resume_from == 'latest':
            model_dir = Path(resume_from).parent
        else:
            model_dir = self.save_dir

        state_path = model_dir / snapshot['state_path']
        assert state_path.exists(), f'State file {state_path} does not exist.'

        return state_path
