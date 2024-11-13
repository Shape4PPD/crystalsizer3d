import shutil
from pathlib import Path
from typing import Optional

import torch
import wx
import yaml
from ruamel.yaml import YAML

from app import ANCHORS_PATH, APP_ASSETS_PATH, APP_DATA_PATH, CRYSTAL_DATA_PATH, DENOISED_IMAGE_PATH, REFINER_ARGS_PATH, \
    SCENE_IMAGE_PATH, SCENE_PATH

from app.components.control_panel import ControlPanel
from app.components.image_panel import ImagePanel
from app.components.optimisation_panel import OptimisationPanel
from app.components.refiner_proxy import RefinerProxy
from app.components.utils import AnchorsChangedEvent, CrystalChangedEvent, CrystalMeshChangedEvent, EVT_ANCHORS_CHANGED, \
    EVT_CRYSTAL_CHANGED, EVT_IMAGE_PATH_CHANGED, EVT_REFINER_ARGS_CHANGED, EVT_SCENE_CHANGED, ImagePathChangedEvent, \
    RefinerArgsChangedEvent, RefinerChangedEvent, SceneChangedEvent
from crystalsizer3d import DATA_PATH, ROOT_PATH, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.projector import Projector
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.util.utils import hash_data


class AppFrame(wx.Frame):
    crystal: Crystal = None
    image_path: Path = None
    refiner_args: Optional[RefinerArgs] = None
    refiner_args_defaults: Optional[RefinerArgs] = None
    refiner: Optional[RefinerProxy] = None
    scene: Optional[Scene] = None
    projector: Optional[Projector] = None

    def __init__(self, config: wx.Config):
        super().__init__(parent=None, title='Crystal Sizer 3D')
        self.config = config
        self.SetMinSize((1200, 600))

        # Initialise panels
        self.control_panel = ControlPanel(self)
        self.image_panel = ImagePanel(self)
        self.optimisation_panel = OptimisationPanel(self)

        # Put panels together
        base_sizer = wx.BoxSizer(wx.HORIZONTAL)
        base_sizer.Add(self.control_panel, 1, wx.EXPAND)
        base_sizer.Add(self.image_panel, 3, wx.EXPAND)
        base_sizer.Add(self.optimisation_panel, 1, wx.EXPAND)
        self.SetSizer(base_sizer)

        # Initialise status bar
        self.CreateStatusBar()
        self.SetStatusText('CrystalSizer3D')

        # Add event listeners
        self.Bind(EVT_IMAGE_PATH_CHANGED, self.update_image_path)
        self.Bind(EVT_CRYSTAL_CHANGED, self.update_crystal)
        self.Bind(EVT_SCENE_CHANGED, self.update_scene)
        self.Bind(EVT_REFINER_ARGS_CHANGED, self.update_refiner_args)
        self.Bind(EVT_ANCHORS_CHANGED, self.update_anchors)

    def _log(self, message: str):
        """
        Log a message to the status bar and to the logger.
        """
        logger.info(message)
        self.SetStatusText(message)

    def load_state(self):
        """
        Load the application state.
        """
        # Load the image(s)
        image_path = self.config.Read('image_path')
        if image_path != '':
            image_path = Path(image_path)
            if image_path.exists():
                self.image_path = image_path
                wx.CallAfter(lambda: wx.PostEvent(self, ImagePathChangedEvent(initial_load=True)))

                # Load the denoised image if it exists
                if DENOISED_IMAGE_PATH.exists():
                    def load_denoised_image():
                        if self.image_panel.images['image'] is None:
                            wx.CallLater(100, load_denoised_image)
                        else:
                            self.image_panel.load_denoised_image(update_images=False)

                    wx.CallAfter(load_denoised_image)

                # Load the scene image if it exists
                if SCENE_IMAGE_PATH.exists():
                    def load_scene_image():
                        if self.image_panel.images['image'] is None:
                            wx.CallLater(100, load_scene_image)
                        else:
                            self.image_panel.load_scene_image(update_images=False)

                    wx.CallAfter(load_scene_image)

        # Instantiate the refiner args
        self.init_refiner_args()

        # Load the scene
        if SCENE_PATH.exists():
            self._ensure_scene_parameters_consistency()
            with open(SCENE_PATH, 'r') as f:
                scene_args = yaml.load(f, Loader=yaml.FullLoader)
            if self.crystal is not None:
                scene_args['crystal'] = self.crystal
            self.scene = Scene.from_dict(scene_args)
            wx.CallAfter(lambda: wx.PostEvent(self, SceneChangedEvent()))

        # Load the crystal
        if CRYSTAL_DATA_PATH.exists():
            self.crystal = Crystal.from_json(CRYSTAL_DATA_PATH)

            # Load any anchors first then update the crystal
            if ANCHORS_PATH.exists():
                with open(ANCHORS_PATH, 'r') as f:
                    anchors = yaml.load(f, Loader=yaml.FullLoader)
                for anchor in anchors:
                    self.image_panel.anchor_manager.anchors[anchor['key']] = torch.tensor(anchor['value'])
                    self.image_panel.anchor_manager.anchor_visibility[anchor['key']] = True
                wx.CallAfter(lambda: wx.PostEvent(self, AnchorsChangedEvent()))

            wx.CallAfter(lambda: wx.PostEvent(self, CrystalChangedEvent(build_mesh=False)))

    def update_crystal(self, event: CrystalChangedEvent):
        """
        Handle crystal updates.
        """
        if self.projector is not None and self.crystal is not None:
            if id(self.projector.crystal) != id(self.crystal):
                self.projector.crystal = self.crystal
        if self.scene is not None and self.crystal is not None:
            if id(self.scene.crystal) != id(self.crystal):
                self.scene.crystal = self.crystal

        # Build the mesh if required
        build_mesh = event.build_mesh if hasattr(event, 'build_mesh') else True
        if build_mesh:
            with torch.no_grad():
                self.crystal.build_mesh()

        # Notify listeners that the crystal mesh has changed
        wx.CallAfter(lambda: wx.PostEvent(self, CrystalMeshChangedEvent()))
        event.Skip()

        # Save the crystal data to file
        if self.crystal is not None:
            self.crystal.to_json(CRYSTAL_DATA_PATH, overwrite=True)

    def update_scene(self, event: wx.Event):
        """
        Handle scene updates.
        """
        # Pull scene updates from the refiner if available
        if self.refiner is not None:
            refiner_scene = self.refiner.scene
            if refiner_scene is not None:
                self.scene = refiner_scene
        if self.crystal is not None and id(self.scene.crystal) != id(self.crystal):
            self.scene.crystal = self.crystal
        if self.projector is not None:
            self.projector.zoom = orthographic_scale_factor(self.scene)
        event.Skip()

        # Save the scene data to file
        if self.scene is not None:
            self.scene.to_yml(SCENE_PATH, overwrite=True)

    def update_image_path(self, event: ImagePathChangedEvent):
        """
        Update the image path references on disk.
        """
        self.config.Write('image_path', str(self.image_path))
        self.config.Flush()
        self.update_refiner_args()
        event.Skip()

    def update_refiner_args(self, event: RefinerArgsChangedEvent = None):
        """
        Update the refiner arguments on disk.
        """
        if self.refiner_args is None:
            self.init_refiner_args()
        yaml = YAML()
        yaml.preserve_quotes = True

        # Ensure the image paths are correct
        self.refiner_args.image_path = self.image_path
        self.refiner_args_defaults.image_path = self.image_path

        # Update just the values present in the refiner args file
        args_updated = False
        with open(REFINER_ARGS_PATH, 'r') as f:
            args_yml = yaml.load(f)
        hash_og = hash_data(args_yml)
        args_dict = self.refiner_args.to_dict()
        for k, v in args_dict.items():
            if k in args_yml:
                args_yml[k] = v
        hash_new = hash_data(args_yml)
        if hash_og != hash_new:
            REFINER_ARGS_PATH.rename(REFINER_ARGS_PATH.with_suffix('.bak'))
            try:
                with open(REFINER_ARGS_PATH, 'w') as f:
                    yaml.dump(args_yml, f)
                REFINER_ARGS_PATH.with_suffix('.bak').unlink()
                self._log('Optimisation settings updated.')
                args_updated = True
            except Exception as e:
                # Restore previous version if there is an error
                REFINER_ARGS_PATH.unlink()
                REFINER_ARGS_PATH.with_suffix('.bak').rename(REFINER_ARGS_PATH)
                self.init_refiner_args()
                self._log('Error updating optimisation settings.')
                logger.error(e)

            self.refiner = None  # This will force the refiner to be reinitialised when next needed

        # Rebuild the scene
        if args_updated:
            self.scene = None
            self.init_scene()

        if event is not None:
            event.Skip()

    def update_anchors(self, event: AnchorsChangedEvent):
        """
        Update the anchors on file.
        """
        event.Skip()
        data = []
        for k, v in self.image_panel.anchor_manager.anchors.items():
            data.append({
                'key': k,
                'value': v.tolist()
            })
        with open(ANCHORS_PATH, 'w') as f:
            yaml.dump(data, f)

    def init_refiner_args(self):
        """
        Initialise the refiner args.
        """
        assert self.image_path is not None
        self._log('Initialising refiner args...')
        yaml = YAML()
        yaml.preserve_quotes = True

        # Load default refiner args
        with open(APP_ASSETS_PATH / 'default_refiner_args.yml') as f:
            default_args_str = f.read()
        default_args_str = default_args_str.replace('%%ROOT_PATH%%', str(ROOT_PATH))
        default_args_str = default_args_str.replace('%%DATA_PATH%%', str(DATA_PATH))
        default_args_yml = yaml.load(default_args_str)
        default_args = RefinerArgs.from_args(default_args_yml)

        # Check if working refiner args exists and are correct
        args_ok = False
        if REFINER_ARGS_PATH.exists():
            with open(REFINER_ARGS_PATH, 'r') as f:
                args_yml = yaml.load(f)
            args_ok = isinstance(args_yml, dict) and all(k in args_yml for k in default_args_yml)

        # Write default refiner args if they don't exist or are broken somehow
        if not args_ok:
            with open(REFINER_ARGS_PATH, 'w') as f:
                f.write(default_args_str)

        # Load the refiner args from file
        with open(REFINER_ARGS_PATH, 'r') as f:
            args_yml = yaml.load(f)
            args = RefinerArgs.from_args(args_yml)

        # Check that the image path is correct
        if self.image_path != args.image_path:
            args.image_path = self.image_path
            for k, v in args.to_dict().items():
                if k in args_yml:
                    args_yml[k] = v
            with open(REFINER_ARGS_PATH, 'w') as f:
                yaml.dump(args_yml, f)

        self.refiner_args = args
        self.refiner_args_defaults = default_args

    def init_refiner(self):
        """
        Initialise the refiner.
        """
        if self.refiner is not None:
            return
        assert self.image_path is not None
        self._log('Initialising refiner...')

        # Instantiate the refiner
        try:
            self.refiner = RefinerProxy(args=self.refiner_args, output_dir=APP_DATA_PATH / 'refiner')
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error initialising refiner',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error initialising refiner.')
            return

        # Notify listeners that the refiner has changed
        wx.PostEvent(self, RefinerChangedEvent())
        wx.PostEvent(self, SceneChangedEvent())

    def _ensure_scene_parameters_consistency(self):
        """
        Ensure that the scene is consistent with the refiner args
        """
        yaml = YAML()
        yaml.preserve_quotes = True

        # Load the scene args from file
        with open(SCENE_PATH, 'r') as f:
            args = yaml.load(f)

        # Check that the scene args are consistent with the refiner args
        args_hash = hash_data(args)
        args['res'] = self.refiner_args.working_image_size
        args['spp'] = self.refiner_args.spp
        args['integrator_max_depth'] = self.refiner_args.integrator_max_depth
        args['integrator_rr_depth'] = self.refiner_args.integrator_rr_depth
        if args_hash != hash_data(args):
            with open(SCENE_PATH, 'w') as f:
                yaml.dump(args, f)

    def init_scene(self):
        """
        Initialise the scene for rendering.
        """
        if self.scene is not None:
            return
        self._log('Initialising scene...')
        yaml = YAML()
        yaml.preserve_quotes = True

        # Copy over default scene if they don't exist
        if not SCENE_PATH.exists():
            shutil.copy(APP_ASSETS_PATH / 'default_scene.yml', SCENE_PATH)

        # Ensure that the scene parameters are consistent with the refiner args
        self._ensure_scene_parameters_consistency()

        # Load the scene args from file
        with open(SCENE_PATH, 'r') as f:
            args = yaml.load(f)
        del args['crystal']

        # Instantiate the scene
        try:
            self.scene = Scene(crystal=self.crystal, **args)
            self._log('Scene initialised.')
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error initialising scene',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error initialising scene.')
            logger.error(e)
            return

    def init_projector(self):
        """
        Initialise the projector for rendering.
        """
        if self.projector is not None:
            return
        if self.scene is None:
            self.init_scene()
            return self.init_projector()
        self._log('Initialising projector...')
        working_image_size = self.config.Read('working_image_size', '800,800')
        image_size = tuple(map(int, working_image_size.split(',')))
        self.projector = Projector(
            crystal=self.crystal,
            image_size=image_size,
            zoom=orthographic_scale_factor(self.scene),
            transparent_background=True
        )
