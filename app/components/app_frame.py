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
    EVT_CRYSTAL_CHANGED, EVT_REFINER_CHANGED, EVT_SCENE_CHANGED, ImagePathChangedEvent, RefinerChangedEvent, \
    SceneChangedEvent
from crystalsizer3d import DATA_PATH, ROOT_PATH, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.projector import Projector
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor


class AppFrame(wx.Frame):
    crystal: Crystal = None
    image_path: Path = None
    refiner: Optional[RefinerProxy] = None
    scene: Optional[Scene] = None
    projector: Optional[Projector] = None

    def __init__(self, config: wx.Config):
        super().__init__(parent=None, title='Crystal Sizer 3D')
        # super().__init__(parent=None, title='Crystal Sizer 3D', size=(1920, 1200))
        self.config = config

        # Disable re-size of the window
        # self.SetWindowStyle(style=wx.DEFAULT_DIALOG_STYLE | wx.MINIMIZE_BOX)

        # Initialise panels
        self.control_panel = ControlPanel(self)
        self.image_panel = ImagePanel(self)
        self.optimisation_panel = OptimisationPanel(self)

        # Put panels together
        base_sizer=wx.BoxSizer(wx.HORIZONTAL)
        base_sizer.Add(self.control_panel, 1, wx.EXPAND)
        base_sizer.Add(self.image_panel, 3, wx.EXPAND)
        base_sizer.Add(self.optimisation_panel, 1, wx.EXPAND)
        self.SetSizer(base_sizer)

        # Initialise status bar
        self.CreateStatusBar()
        self.SetStatusText('CrystalSizer3D')

        # Add event listeners
        self.Bind(EVT_CRYSTAL_CHANGED, self.update_crystal)
        self.Bind(EVT_SCENE_CHANGED, self.update_scene)
        self.Bind(EVT_REFINER_CHANGED, self.update_refiner)
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

        if SCENE_PATH.exists():
            with open(SCENE_PATH, 'r') as f:
                scene_args = yaml.load(f, Loader=yaml.FullLoader)
            if self.crystal is not None:
                scene_args['crystal'] = self.crystal
            self.scene = Scene.from_dict(scene_args)
            wx.CallAfter(lambda: wx.PostEvent(self, SceneChangedEvent()))

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

    def update_refiner(self, event: wx.Event):
        """
        Update the refiner.
        """
        pass

    def update_anchors(self, event: wx.Event):
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

    def init_refiner(self):
        """
        Initialise the refiner.
        """
        if self.refiner is not None:
            return
        assert self.image_path is not None
        self._log('Initialising refiner...')
        yaml = YAML()
        yaml.preserve_quotes = True

        # Copy over default refiner args if they don't exist
        if not REFINER_ARGS_PATH.exists():
            with open(APP_ASSETS_PATH / 'default_refiner_args.yml') as f:
                default_args = f.read()
            default_args = default_args.replace('%%ROOT_PATH%%', str(ROOT_PATH))
            default_args = default_args.replace('%%DATA_PATH%%', str(DATA_PATH))
            with open(REFINER_ARGS_PATH, 'w') as f:
                f.write(default_args)

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

        # Instantiate the refiner
        try:
            self.refiner = RefinerProxy(args=args, output_dir=APP_DATA_PATH / 'refiner')
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error initialising refiner',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error initialising refiner.')
            return

        wx.PostEvent(self, RefinerChangedEvent())
        wx.PostEvent(self, SceneChangedEvent())

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

        # Load the scene args from file
        with open(SCENE_PATH, 'r') as f:
            args = yaml.load(f)

        # Instantiate the scene
        try:
            working_image_size = self.config.Read('working_image_size', '300,300')
            image_size = tuple(map(int, working_image_size.split(',')))
            self.scene = Scene(
                crystal=self.crystal,
                res=min(image_size),
                **args
            )
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error initialising scene',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error initialising scene.')
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
        working_image_size = self.config.Read('working_image_size', '300,300')
        image_size = tuple(map(int, working_image_size.split(',')))
        self.projector = Projector(
            crystal=self.crystal,
            image_size=image_size,
            zoom=orthographic_scale_factor(self.scene),
            transparent_background=True
        )
