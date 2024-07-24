import wx
from ruamel.yaml import YAML

from app import APP_ASSETS_PATH, APP_DATA_PATH, REFINER_ARGS_PATH
from app.components.app_panel import AppPanel
from app.components.utils import CrystalChangedEvent, DenoisedImageChangedEvent, EVT_CRYSTAL_CHANGED, \
    EVT_IMAGE_PATH_CHANGED, ImagePathChangedEvent, RefinerChangedEvent, SceneImageChangedEvent
from crystalsizer3d import DATA_PATH, ROOT_PATH, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.refiner.refiner import Refiner


class OptimisationPanel(AppPanel):
    @property
    def refiner(self):
        return self.app_frame.refiner

    def _init_components(self):
        """
        Initialise the optimisation panel components.
        """
        self.title = wx.StaticText(self, label='Mesh Optimiser')

        # Initial prediction
        self.btn_initial_prediction = wx.Button(self, label='Make Initial Prediction')
        self.btn_initial_prediction.Bind(wx.EVT_BUTTON, self.make_initial_prediction)

        # Constraints list
        self.constraint_list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.constraint_list.SetMinSize(wx.Size(256, 256))
        self.constraint_list.InsertColumn(0, 'Vertex')
        # self.constraint_list.SetColumnWidth(col=0, width=100)
        self.constraint_list.InsertColumn(1, '2D Coordinate')
        # self.constraint_list.SetColumnWidth(col=1, width=100)

        # Refine solution button
        self.btn_refine = wx.Button(self, label='Refine')
        self.btn_refine.Bind(wx.EVT_BUTTON, self.refine_prediction)

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.title, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        main_sizer.Add(self.btn_initial_prediction, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.constraint_list, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.btn_refine, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(main_sizer)

    def _init_listeners(self):
        """
        Initialise the event listeners.
        """
        self.app_frame.Bind(EVT_IMAGE_PATH_CHANGED, self.image_changed)

    def image_changed(self, event: ImagePathChangedEvent):
        """
        Image has changed, so we need a new refiner.
        """
        self.app_frame.refiner = None
        wx.PostEvent(self.app_frame, RefinerChangedEvent())
        event.Skip()

    def _init_refiner(self):
        """
        Initialise the refiner.
        """
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
            self.app_frame.refiner = Refiner(args=args, output_dir=APP_DATA_PATH / 'refiner')
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error initialising refiner',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error initialising refiner.')
            return

        wx.PostEvent(self.app_frame, RefinerChangedEvent())

    def make_initial_prediction(self, event: wx.Event):
        """
        Get the initial crystal prediction using a trained neural network predictor model.
        """
        if self.image_path is None:
            wx.MessageBox(message='You must load an image first.', caption='CrystalSizer3D',
                          style=wx.OK | wx.ICON_ERROR)
            return
        self._log('Getting initial prediction...')
        if self.refiner is None:
            self._init_refiner()
            if self.refiner is None:
                return
        try:
            self.refiner.make_initial_prediction()
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error making initial prediction',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error making initial prediction.')
            logger.error(str(e))
            return
        self.app_frame.crystal = self.refiner.crystal
        wx.PostEvent(self.app_frame, CrystalChangedEvent())
        wx.PostEvent(self.app_frame, DenoisedImageChangedEvent())
        wx.PostEvent(self.app_frame, SceneImageChangedEvent())
        self._log('Initial prediction complete.')

    def refine_prediction(self, event: wx.Event):
        """
        Refine the prediction.
        """
        event.Skip()
        if self.refiner is None:
            self._init_refiner()

        # If the refiner has no crystal, use the current crystal or make an initial prediction
        if self.refiner.crystal is None:
            if self.crystal is None:
                self.make_initial_prediction(event)
            else:
                self.refiner.crystal = self.crystal

        # Callback
        def after_refine_step():
            wx.PostEvent(self.app_frame, CrystalChangedEvent())
            wx.PostEvent(self.app_frame, SceneImageChangedEvent())

        # Refine the prediction
        self._log('Refining prediction...')
        self.refiner.train(after_refine_step)
        after_refine_step()
        self._log('Prediction refined.')
