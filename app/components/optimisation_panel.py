import threading
import time

import wx

from app.components.app_panel import AppPanel
from app.components.parallelism import start_thread, stop_event
from app.components.utils import CrystalChangedEvent, DenoisedImageChangedEvent, EVT_IMAGE_PATH_CHANGED, \
    ImagePathChangedEvent, RefinerChangedEvent, SceneChangedEvent, SceneImageChangedEvent
from crystalsizer3d import logger


class OptimisationPanel(AppPanel):
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

    def make_initial_prediction(self, event: wx.Event):
        """
        Get the initial crystal prediction using a trained neural network predictor model.
        """
        if self.image_path is None:
            wx.MessageBox(message='You must load an image first.', caption='CrystalSizer3D',
                          style=wx.OK | wx.ICON_ERROR)
            return
        self._log('Getting initial prediction...')
        try:
            self.refiner.make_initial_prediction()
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error making initial prediction',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error making initial prediction.')
            logger.error(str(e))
            return
        self.app_frame.crystal = self.refiner.crystal
        wx.PostEvent(self.app_frame, SceneChangedEvent())
        wx.PostEvent(self.app_frame, DenoisedImageChangedEvent())
        wx.PostEvent(self.app_frame, SceneImageChangedEvent())
        self._log('Sending crystal changed event.')
        wx.CallAfter(wx.PostEvent, self.app_frame, CrystalChangedEvent(build_mesh=False))
        self._log('Initial prediction complete.')

    def refine_prediction(self, event: wx.Event):
        """
        Refine the prediction.
        """
        event.Skip()
        if self.refiner.is_training():
            self.refiner.stop_training()
            event.EventObject.SetLabel('Refine')
            self.Refresh()
            wx.Yield()
            return
        event.EventObject.SetLabel('Stop refining')
        self.Refresh()
        wx.Yield()

        # If the refiner has no crystal, use the current crystal or make an initial prediction
        if self.refiner.crystal is None:
            if self.crystal is None:
                self.make_initial_prediction(event)
            else:
                # self.refiner.set_initial_data(
                #     crystal=self.crystal,
                # )
                # self.refiner.crystal = self.crystal
                pass

        # Callback
        def after_refine_step(step: int):
            if step % 5 != 0:
                return
            wx.CallAfter(wx.PostEvent, self.app_frame, CrystalChangedEvent(build_mesh=False))
            wx.CallAfter(wx.PostEvent, self.app_frame, SceneImageChangedEvent())

        # Refine the prediction
        def training_thread(stop_event: threading.Event):
            self._log('Refining prediction...')
            self.refiner.train(after_refine_step)
            while self.refiner.is_training() and not stop_event.is_set():
                time.sleep(1)
            after_refine_step(self.refiner.step)
            event.EventObject.SetLabel('Refine')
            self._log('Prediction refined.')

        thread = threading.Thread(target=training_thread, args=(stop_event,))
        start_thread(thread)
