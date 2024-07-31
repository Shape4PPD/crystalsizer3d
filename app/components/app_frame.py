import shutil
from pathlib import Path
from typing import Optional

import torch
import wx
import yaml
from ruamel.yaml import YAML

from app import APP_ASSETS_PATH, APP_DATA_PATH, CRYSTAL_DATA_PATH, DENOISED_IMAGE_PATH, REFINER_ARGS_PATH, \
    SCENE_IMAGE_PATH, SCENE_PATH
from app.components.control_panel import ControlPanel
from app.components.image_panel import ImagePanel
from app.components.optimisation_panel import OptimisationPanel
from app.components.refiner_proxy import RefinerProxy
from app.components.utils import CrystalChangedEvent, CrystalMeshChangedEvent, EVT_CRYSTAL_CHANGED, EVT_REFINER_CHANGED, \
    EVT_SCENE_CHANGED, ImagePathChangedEvent, RefinerChangedEvent, SceneChangedEvent
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
        super().__init__(parent=None, title='Crystal Sizer 3D', size=(1280, 720))
        self.config = config
        self.SetWindowStyle(style=wx.DEFAULT_DIALOG_STYLE | wx.MINIMIZE_BOX)

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
        self.Bind(EVT_CRYSTAL_CHANGED, self.update_crystal)
        self.Bind(EVT_SCENE_CHANGED, self.update_scene)
        self.Bind(EVT_REFINER_CHANGED, self.update_refiner)

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
                        if self.image_panel.image is None:
                            wx.CallLater(100, load_denoised_image)
                        else:
                            self.image_panel.load_denoised_image()

                    wx.CallAfter(load_denoised_image)

                # Load the scene image if it exists
                if SCENE_IMAGE_PATH.exists():
                    def load_scene_image():
                        if self.image_panel.image is None:
                            wx.CallLater(100, load_scene_image)
                        else:
                            self.image_panel.load_scene_image()

                    wx.CallAfter(load_scene_image)

        if CRYSTAL_DATA_PATH.exists():
            self.crystal = Crystal.from_json(CRYSTAL_DATA_PATH)
            wx.CallAfter(lambda: wx.PostEvent(self, CrystalChangedEvent(build_mesh=False)))

        if SCENE_PATH.exists():
            with open(SCENE_PATH, 'r') as f:
                scene_args = yaml.load(f, Loader=yaml.FullLoader)
            if self.crystal is not None:
                scene_args['crystal'] = self.crystal
            self.scene = Scene.from_dict(scene_args)
            wx.CallAfter(lambda: wx.PostEvent(self, SceneChangedEvent()))

    def update_crystal(self, event: CrystalChangedEvent):
        """
        Handle crystal updates.
        """
        # Pull crystal updates from the refiner if available
        if self.refiner is not None:
            refiner_crystal = self.refiner.crystal
            if refiner_crystal is not None:
                self.crystal = refiner_crystal
        if self.projector is not None and self.crystal is not None:
            if id(self.projector.crystal) != id(self.crystal):
                self.projector.crystal = self.crystal
        event.Skip()

        # Build the mesh if required
        build_mesh = event.build_mesh if hasattr(event, 'build_mesh') else True
        if build_mesh:
            with torch.no_grad():
                self.crystal.build_mesh()

        # Notify listeners that the crystal mesh has changed
        wx.CallAfter(lambda: wx.PostEvent(self, CrystalMeshChangedEvent()))

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

    def on_optim(self, event):
        print('> vertices')
        print(self.crystal.vertices)
        print(self.vertices_2d)
        print('> faces')
        print(self.crystal.faces)

        # # One time use only - generate target positions from fitted crystal
        # # 24/06/24 - ImageBitmap size change - Remember to redo all coordinates!
        # dctlst = []
        # for i in range(len(self.crystal.vertices)):
        #     fcslst = []
        #     for mlr_idx in self.crystal.faces:
        #         if i in self.crystal.faces[mlr_idx].tolist():
        #             fcslst.append(mlr_idx)
        #     dctlst.append({'faces': fcslst, 'position_2d': self.vertices_2d[i].tolist()})
        # print(dctlst)

        # Fitted crystal - alpha4.json
        target_positions = [
            {'faces': [(0, 0, 1), (1, 1, 1), (1, -1, 1)], 'position_2d': [460.37896728515625, 409.05523681640625]},
            {'faces': [(0, 0, 1), (1, 1, 1), (0, 1, 1)], 'position_2d': [344.228759765625, 360.9009704589844]},
            {'faces': [(0, 0, 1), (1, -1, 1), (0, -1, 1)], 'position_2d': [542.4766845703125, 322.4871520996094]},
            {'faces': [(0, 0, 1), (-1, 1, 1), (-1, -1, 1)], 'position_2d': [412.1661376953125, 260.7255859375]},
            {'faces': [(0, 0, 1), (-1, 1, 1), (0, 1, 1)], 'position_2d': [339.2733154296875, 337.5874938964844]},
            {'faces': [(0, 0, 1), (-1, -1, 1), (0, -1, 1)], 'position_2d': [540.6732788085938, 314.00286865234375]},
            {'faces': [(0, 0, -1), (1, 1, -1), (1, -1, -1)], 'position_2d': [456.6241760253906, 419.4239807128906]},
            {'faces': [(0, 0, -1), (1, 1, -1), (0, 1, -1)], 'position_2d': [351.35150146484375, 375.77935791015625]},
            {'faces': [(0, 0, -1), (1, -1, -1), (0, -1, -1)], 'position_2d': [546.0210571289062, 325.1593017578125]},
            {'faces': [(0, 0, -1), (-1, 1, -1), (-1, -1, -1)], 'position_2d': [425.3760681152344, 249.98631286621094]},
            {'faces': [(0, 0, -1), (-1, 1, -1), (0, 1, -1)], 'position_2d': [343.0623779296875, 336.7821044921875]},
            {'faces': [(0, 0, -1), (-1, -1, -1), (0, -1, -1)], 'position_2d': [540.1574096679688, 297.5730895996094]},
            {'faces': [(1, 1, 1), (1, 1, -1), (1, 0, 0)], 'position_2d': [452.66888427734375, 429.6434631347656]},
            {'faces': [(1, 1, 1), (1, 1, -1), (0, 1, -1)], 'position_2d': [332.44146728515625, 379.7987976074219]},
            {'faces': [(1, 1, 1), (1, -1, 1), (1, 0, 0)], 'position_2d': [464.232666015625, 427.18548583984375]},
            {'faces': [(1, 1, 1), (0, 1, 1), (0, 1, -1)], 'position_2d': [317.7055358886719, 366.5386657714844]},
            {'faces': [(1, 1, -1), (1, -1, -1), (1, 0, 0)], 'position_2d': [458.53155517578125, 428.3973083496094]},
            {'faces': [(1, -1, 1), (1, -1, -1), (1, 0, 0)], 'position_2d': [470.0953063964844, 425.9393615722656]},
            {'faces': [(1, -1, 1), (1, -1, -1), (0, -1, -1)], 'position_2d': [570.6322021484375, 319.92803955078125]},
            {'faces': [(1, -1, 1), (0, -1, 1), (0, -1, -1)], 'position_2d': [572.3182983398438, 316.1440734863281]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (-1, 0, 0)], 'position_2d': [405.86444091796875, 245.04063415527344]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (0, 1, 1), (0, 1, -1)],
             'position_2d': [312.7500915527344, 343.2251892089844]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (0, 1, 1), (0, 1, -1)],
             'position_2d': [312.7500915527344, 343.2251892089844]},
            {'faces': [(-1, 1, 1), (-1, -1, 1), (-1, 0, 0)], 'position_2d': [408.7038879394531, 244.43707275390625]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (0, 1, 1), (0, 1, -1)],
             'position_2d': [312.7500915527344, 343.2251892089844]},
            {'faces': [(-1, 1, -1), (-1, -1, -1), (-1, 0, 0)], 'position_2d': [423.5268249511719, 241.286376953125]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (0, 1, 1), (0, 1, -1)],
             'position_2d': [312.7500915527344, 343.2251892089844]},
            {'faces': [(-1, -1, 1), (-1, -1, -1), (-1, 0, 0)], 'position_2d': [426.3663024902344, 240.6828155517578]},
            {'faces': [(-1, -1, 1), (-1, -1, -1), (0, -1, -1)], 'position_2d': [555.646728515625, 294.2807312011719]},
            {'faces': [(-1, -1, 1), (0, -1, 1), (0, -1, -1)], 'position_2d': [570.514892578125, 307.6598815917969]}]

        # Optimizer from Tom I
        opt = torch.optim.Adam([
            self.crystal.distances
        ], lr=0.01)

        for i in range(100):
            opt.zero_grad()
            self.update_crystal(None)
            # self.crystal.update_mesh()
            loss = 0
            for target_position in target_positions:

                # Find the joining vertex from faces - index is 'x'
                # Continue if not found or find more than 1
                xf = list(range(0, len(self.crystal.vertices)))
                for face in target_position['faces']:
                    if face in self.crystal.faces:
                        xf = list(set(xf).intersection(set(self.crystal.faces[face].tolist())))
                    else:
                        continue
                if len(xf) != 1:
                    continue
                x = xf[0]

                loss = loss + torch.norm(
                    self.vertices_2d[x] - torch.tensor(target_position['position_2d'])
                )
            loss.backward()
            print(f'Step {i}, Loss: {loss.item()}, Distances: {self.crystal.distances}')

            # Update distances in list
            for t in range(0, len(self.crystal.distances)):
                self.face_list.SetItem(t, 1, str(format(self.crystal.distances[t], '.2f')))

            # Save image of every step
            filename = 'Step_' + '{:03d}'.format(i) + '.jpg'
            bitmap = self.img_bitmap.GetBitmap()
            bitmap.SaveFile('../TestData_Optimizer_2/demo/optimSteps/' + filename, wx.BITMAP_TYPE_JPEG)

            opt.step()
