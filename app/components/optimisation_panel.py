import threading
import time
from typing import Dict, Optional, Tuple, Union

import wx
from torch import Tensor

from app.components.anchor_manager import AnchorManager
from app.components.app_panel import AppPanel
from app.components.parallelism import start_thread, stop_event
from app.components.utils import CrystalChangedEvent, DenoisedImageChangedEvent, EVT_ANCHORS_CHANGED, \
    EVT_IMAGE_PATH_CHANGED, EVT_REFINING_STARTED, ImagePathChangedEvent, RefinerChangedEvent, RefiningEndedEvent, \
    RefiningStartedEvent, SceneImageChangedEvent
from crystalsizer3d import logger


class OptimisationPanel(AppPanel):
    selected_anchor_idx: Optional[int] = None
    step: int = 0

    @property
    def anchor_manager(self) -> AnchorManager:
        return self.app_frame.image_panel.anchor_manager

    @property
    def anchors(self) -> Dict[Tuple[Union[str, int], int], Tensor]:
        return self.anchor_manager.anchors

    def _init_components(self):
        """
        Initialise the optimisation panel components.
        """
        self.title = wx.StaticText(self, label='Mesh Optimiser')

        # Initial prediction
        self.btn_initial_prediction = wx.Button(self, label='Make Initial Prediction')
        self.btn_initial_prediction.Bind(wx.EVT_BUTTON, self.make_initial_prediction)

        # Anchors list
        self.anchors_list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.anchors_list.SetMinSize(wx.Size(256, 256))
        self.anchors_list.InsertColumn(0, 'Vertex')
        self.anchors_list.SetColumnWidth(col=0, width=80)
        self.anchors_list.InsertColumn(1, 'In face')
        self.anchors_list.SetColumnWidth(col=1, width=80)
        self.anchors_list.InsertColumn(2, '2D Coordinate')
        self.anchors_list.SetColumnWidth(col=2, width=100)
        self.anchors_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_anchor_select)
        self.anchors_list.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.on_anchor_deselect)
        self.anchors_list.Bind(wx.EVT_LEFT_DOWN, self.on_anchor_list_click)

        # Anchors buttons
        self.btn_remove_anchor = wx.Button(self, label='Remove')
        self.btn_remove_anchor.Enable(False)
        self.btn_remove_anchor.Bind(wx.EVT_BUTTON, self.remove_selected_anchor)
        self.btn_remove_all_anchors = wx.Button(self, label='Remove All')
        self.btn_remove_all_anchors.Enable(False)
        self.btn_remove_all_anchors.Bind(wx.EVT_BUTTON, self.remove_all_anchors)
        anchor_btns_sizer = wx.BoxSizer(wx.HORIZONTAL)
        anchor_btns_sizer.Add(self.btn_remove_anchor, 1, wx.EXPAND)
        anchor_btns_sizer.Add(self.btn_remove_all_anchors, 1, wx.EXPAND)

        # Anchors sizer
        anchors_sizer = wx.StaticBoxSizer(wx.VERTICAL, parent=self, label='Anchors')
        anchors_sizer.Add(self.anchors_list, 0, wx.EXPAND | wx.ALL, 3)
        anchors_sizer.Add(anchor_btns_sizer, 0, wx.EXPAND | wx.ALL, 3)

        # Refine solution button
        self.btn_refine = wx.Button(self, label='Refine')
        self.btn_refine.Bind(wx.EVT_BUTTON, self.refine_prediction)

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.title, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        main_sizer.Add(self.btn_initial_prediction, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(anchors_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.btn_refine, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(main_sizer)

    def _init_listeners(self):
        """
        Initialise the event listeners.
        """
        self.app_frame.Bind(EVT_IMAGE_PATH_CHANGED, self.image_changed)
        self.app_frame.Bind(EVT_ANCHORS_CHANGED, self.anchors_changed)
        self.app_frame.Bind(EVT_REFINING_STARTED, self.monitor_refining_updates)

    def image_changed(self, event: ImagePathChangedEvent):
        """
        Image has changed, so we need a new refiner.
        """
        self.app_frame.refiner = None
        wx.PostEvent(self.app_frame, RefinerChangedEvent())
        event.Skip()

    def anchors_changed(self, event: wx.Event):
        """
        Anchors have changed.
        """
        event.Skip()
        self.anchors_list.DeleteAllItems()
        for i, (vertex_key, coord) in enumerate(self.anchors.items()):
            vertex_id, face_idx = vertex_key
            self.anchors_list.InsertItem(i, str(vertex_id))
            self.anchors_list.SetItem(i, 1, str(face_idx))
            self.anchors_list.SetItem(i, 2, f'({coord[0]:.2f}, {coord[1]:.2f})')
            if not self.anchor_manager.anchor_visibility[vertex_key]:
                self.anchors_list.SetItemTextColour(i, wx.Colour(150, 50, 50))
        self.btn_remove_anchor.Disable()
        is_training = 'refiner' in self.__dict__ and self.refiner.is_training()
        self.btn_remove_all_anchors.Enable(len(self.anchors) > 0 and not is_training)

    def on_anchor_select(self, event: wx.Event):
        """
        Select an anchor.
        """
        event.Skip()
        idx = self.anchors_list.GetFirstSelected()
        if idx == self.selected_anchor_idx:
            self.anchors_list.SetItemState(idx, 0, wx.LIST_STATE_SELECTED)
            idx = None
        if idx == -1:
            idx = None
        self.selected_anchor_idx = idx
        self.anchor_manager.select_anchor(self.selected_anchor_idx)
        self.btn_remove_anchor.Enable(self.selected_anchor_idx is not None)

    def on_anchor_deselect(self, event: wx.Event):
        """
        Deselect an anchor.
        """
        event.Skip()
        self.selected_anchor_idx = None
        self.app_frame.image_panel.anchor_manager.select_anchor(self.selected_anchor_idx)
        self.btn_remove_anchor.Disable()

    def on_anchor_list_click(self, event: wx.MouseEvent):
        """
        Toggle the selection of an anchor - if it is already selected, deselect it.
        """
        x, y = event.GetPosition()
        item, flags = self.anchors_list.HitTest((x, y))
        if item == self.selected_anchor_idx:
            self.anchors_list.SetItemState(item, 0, wx.LIST_STATE_SELECTED)
            self.selected_anchor_idx = None
            self.anchor_manager.select_anchor(self.selected_anchor_idx)
            self.btn_remove_anchor.Disable()
        else:
            event.Skip()

    def remove_selected_anchor(self, event: wx.Event):
        """
        Remove the selected anchor.
        """
        event.Skip()
        if self.selected_anchor_idx is None:
            return
        self.anchor_manager.remove_anchor(self.selected_anchor_idx)

    def remove_all_anchors(self, event: wx.Event):
        """
        Remove all anchors.
        """
        event.Skip()
        self.anchor_manager.remove_all_anchors()

    def make_initial_prediction(self, event: wx.Event):
        """
        Get the initial crystal prediction using a trained neural network predictor model.
        """
        if self.image_path is None:
            wx.MessageBox(message='You must load an image first.', caption='CrystalSizer3D',
                          style=wx.OK | wx.ICON_ERROR)
            return
        self._log('Generating initial prediction...')
        wx.PostEvent(self.app_frame, RefiningStartedEvent())
        try:
            self.refiner.make_initial_prediction()
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error making initial prediction',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error making initial prediction.')
            logger.error(str(e))
            return
        self.app_frame.crystal = self.refiner.crystal
        wx.PostEvent(self.app_frame, RefiningEndedEvent())
        wx.PostEvent(self.app_frame, DenoisedImageChangedEvent())
        wx.PostEvent(self.app_frame, SceneImageChangedEvent(update_images=False))
        wx.CallAfter(wx.PostEvent, self.app_frame, CrystalChangedEvent(build_mesh=False))
        self._log('Initial prediction complete.')

    def refine_prediction(self, event: wx.Event):
        """
        Refine the prediction.
        """
        event.Skip()

        # If the refiner is already training, send the stop signal
        if self.refiner.is_training():
            self.btn_refine.SetLabel('Stopping...')
            self.Refresh()
            self.refiner.stop_training()
            return

        # Check that there is a crystal to refine
        if self.crystal is None:
            wx.MessageBox(message='First generate an initial prediction or load a saved crystal from file.',
                          caption='No crystal shape to refine.',
                          style=wx.OK | wx.ICON_ERROR)
            return

        def after_refine_step(step: int):
            self.step = step

        # Refine the prediction
        def training_thread(stop_event: threading.Event):
            self._log('Refining prediction...')

            # Set the anchors and initial scene
            self.refiner.set_anchors(self.anchors)
            self.refiner.set_initial_scene(self.scene)

            # Run the training
            wx.CallAfter(wx.PostEvent, self.app_frame, RefiningStartedEvent())
            self.refiner.train(after_refine_step)
            while self.refiner.is_training() and not stop_event.is_set():
                time.sleep(1)
            self._log('Refining prediction complete.')
            self._log(f'Stopped at step = {self.refiner.step}')

            # Send events and update buttons
            self.step = self.refiner.step
            wx.PostEvent(self.app_frame, RefiningEndedEvent())
            self.btn_refine.SetLabel('Refine')
            self.btn_initial_prediction.Enable()
            self.btn_remove_all_anchors.Enable(len(self.anchors) > 0)
            self._log('Prediction refined.')

        # Disable the other action buttons
        self.btn_initial_prediction.Disable()
        self.btn_remove_anchor.Disable()
        self.btn_remove_all_anchors.Disable()
        self.btn_refine.SetLabel('Stop refining')
        self.Refresh()

        # Start refining - run in a separate thread to keep the UI responsive
        thread = threading.Thread(target=training_thread, args=(stop_event,))
        start_thread(thread)

    def monitor_refining_updates(self, event: RefiningStartedEvent):
        """
        Monitor the refining updates.
        """
        event.Skip()
        sleep_time = int(self.config.Read('refining_update_ui_every_n_seconds', '2'))

        def update_ui_thread(stop_event: threading.Event):
            while self.refiner.is_training() and not stop_event.is_set():
                if self.app_frame.image_panel.images_updating:
                    time.sleep(2)
                    continue
                self.app_frame.crystal = self.refiner.crystal  # Update the crystal
                wx.PostEvent(self.app_frame, CrystalChangedEvent(build_mesh=False))
                wx.PostEvent(self.app_frame, SceneImageChangedEvent(update_images=False))
                time.sleep(sleep_time)

        thread = threading.Thread(target=update_ui_thread, args=(stop_event,))
        start_thread(thread)
