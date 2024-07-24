from pathlib import Path

import numpy as np
import torch
import wx

from app.components.app_panel import AppPanel
from app.components.utils import CrystalChangedEvent, EVT_CRYSTAL_CHANGED, ImagePathChangedEvent
from crystalsizer3d import logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.utils import to_numpy


class ControlPanel(AppPanel):
    STEP_ROTATION = 0.1
    STEP_TRANSLATION = 0.1
    STEP_DISTANCE = 0.02
    STEP_SCALE = 0.05
    STEP_IOR = 0.1

    def _init_components(self):
        """
        Initialise the control panel components.
        """
        self.title = wx.StaticText(self, label='Crystal Sizer 3D (v0.0.1)')

        # Panel for file control
        self.btn_load_image = wx.Button(self, label='Load Image...')
        self.btn_load_image.Bind(wx.EVT_BUTTON, self.on_load_image)
        self.btn_load_crystal = wx.Button(self, label='Load Crystal...')
        self.btn_load_crystal.Bind(wx.EVT_BUTTON, self.on_load_crystal)
        self.btn_save_crystal = wx.Button(self, label='Save Crystal...')
        self.btn_save_crystal.Bind(wx.EVT_BUTTON, self.on_save_crystal)
        fc_sizer = wx.StaticBoxSizer(wx.VERTICAL, parent=self, label='File Control')
        fc_sizer.Add(self.btn_load_image, 0, wx.EXPAND | wx.ALL, 3)
        fc_sizer.Add(self.btn_load_crystal, 0, wx.EXPAND | wx.ALL, 3)
        fc_sizer.Add(self.btn_save_crystal, 0, wx.EXPAND | wx.ALL, 3)

        # Face list
        self.face_list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.face_list.SetMinSize(wx.Size(256, 256))
        self.face_list.InsertColumn(0, 'Miller Index')
        self.face_list.SetColumnWidth(col=0, width=100)
        self.face_list.InsertColumn(1, 'Distance')
        self.face_list.SetColumnWidth(col=1, width=100)

        # Panel for adjusting face normal distances
        self.btn_distance_up = wx.Button(self, label='Dis. +')
        self.btn_distance_up.Bind(wx.EVT_BUTTON, self.on_distance_change)
        self.btn_distance_down = wx.Button(self, label='Dis. -')
        self.btn_distance_down.Bind(wx.EVT_BUTTON, self.on_distance_change)
        d_sizer = wx.BoxSizer(wx.HORIZONTAL)
        d_sizer.Add(self.btn_distance_up, 1, wx.EXPAND)
        d_sizer.Add(self.btn_distance_down, 1, wx.EXPAND)

        # Panel for adjusting scale (normal distances * scale param)
        self.btn_scale_up = wx.Button(self, label='Scl. +')
        self.btn_scale_up.Bind(wx.EVT_BUTTON, self.on_scale_change)
        self.btn_scale_down = wx.Button(self, label='Scl. -')
        self.btn_scale_down.Bind(wx.EVT_BUTTON, self.on_scale_change)
        s_sizer = wx.BoxSizer(wx.HORIZONTAL)
        s_sizer.Add(self.btn_scale_up, 1, wx.EXPAND)
        s_sizer.Add(self.btn_scale_down, 1, wx.EXPAND)

        # Panel for controlling crystal rotation and position (transformation)
        self.btn_rotate_cw = wx.Button(self, label='↻')
        self.btn_rotate_cw.Bind(wx.EVT_BUTTON, self.on_rotate)
        self.btn_rotate_ccw = wx.Button(self, label='↺')
        self.btn_rotate_ccw.Bind(wx.EVT_BUTTON, self.on_rotate)
        self.btn_translate_up = wx.Button(self, label='↑')
        self.btn_translate_up.Bind(wx.EVT_BUTTON, self.on_translate)
        self.btn_translate_left = wx.Button(self, label='←')
        self.btn_translate_left.Bind(wx.EVT_BUTTON, self.on_translate)
        self.btn_translate_down = wx.Button(self, label='↓')
        self.btn_translate_down.Bind(wx.EVT_BUTTON, self.on_translate)
        self.btn_translate_right = wx.Button(self, label='→')
        self.btn_translate_right.Bind(wx.EVT_BUTTON, self.on_translate)

        # Rotation and Position text
        self.lbl_rotation = wx.StaticText(self, label='Rotation: 0°')
        self.lbl_position = wx.StaticText(self, label='Position: 0, 0')

        t_sizer_top = wx.BoxSizer(wx.HORIZONTAL)
        t_sizer_top.Add(self.btn_rotate_cw, 1, wx.EXPAND)
        t_sizer_top.Add(self.btn_translate_up, 1, wx.EXPAND)
        t_sizer_top.Add(self.btn_rotate_ccw, 1, wx.EXPAND)
        t_sizer_bottom = wx.BoxSizer(wx.HORIZONTAL)
        t_sizer_bottom.Add(self.btn_translate_left, 1, wx.EXPAND)
        t_sizer_bottom.Add(self.btn_translate_down, 1, wx.EXPAND)
        t_sizer_bottom.Add(self.btn_translate_right, 1, wx.EXPAND)
        l_sizer = wx.BoxSizer(wx.HORIZONTAL)
        l_sizer.Add(self.lbl_rotation, 1, wx.EXPAND)
        l_sizer.Add(self.lbl_position, 1, wx.EXPAND)
        t_sizer = wx.StaticBoxSizer(wx.VERTICAL, parent=self, label='Transformation')
        t_sizer.Add(t_sizer_top, 0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, 5)
        t_sizer.Add(t_sizer_bottom, 0, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 5)
        t_sizer.Add(l_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # Panel for refractive index
        self.lbl_ior = wx.StaticText(self, label='IOR: 1', style=wx.ALIGN_CENTER_HORIZONTAL)
        self.btn_ior_up = wx.Button(self, label='+')
        self.btn_ior_up.Bind(wx.EVT_BUTTON, self.on_ior_change)
        self.btn_ior_down = wx.Button(self, label='-')
        self.btn_ior_down.Bind(wx.EVT_BUTTON, self.on_ior_change)
        ior_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ior_sizer.Add(self.lbl_ior, 1, wx.EXPAND | wx.ALL, 3)
        ior_sizer.Add(self.btn_ior_up, 1, wx.EXPAND | wx.ALL, 3)
        ior_sizer.Add(self.btn_ior_down, 1, wx.EXPAND | wx.ALL, 3)

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.title, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        main_sizer.Add(fc_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.face_list, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(d_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        main_sizer.Add(s_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        main_sizer.Add(t_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(ior_sizer, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(main_sizer)

    def _init_listeners(self):
        """
        Initialise the event listeners.
        """
        self.app_frame.Bind(EVT_CRYSTAL_CHANGED, self.update_face_list)
        self.app_frame.Bind(EVT_CRYSTAL_CHANGED, self.update_control_labels)

    def on_load_image(self, event):
        """
        Show the file dialog to load an image.
        """
        loadImgDiag = wx.FileDialog(self, message='Load Image', wildcard='Images|*.jpg;*.png',
                                    style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if loadImgDiag.ShowModal() == wx.ID_CANCEL:
            return
        filepath = loadImgDiag.GetPath()
        try:
            image = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
            assert image.IsOk(), 'Invalid image file'
            self.app_frame.image_path = filepath
            self.config.Write('image_path', filepath)
            self.config.Flush()
            wx.PostEvent(self.app_frame, ImagePathChangedEvent())
        except Exception as e:
            self._log('Loading image failed.')
            logger.error(f'Loading image failed: {e}')
            wx.MessageBox(message=str(e), caption='Error', style=wx.OK | wx.ICON_ERROR)
        loadImgDiag.Destroy()

    def on_load_crystal(self, event: wx.Event):
        """
        Load a crystal from json file.
        """
        wildcard = 'Crystal|*.json'
        dialog = wx.FileDialog(self, message='Load Crystal', wildcard=wildcard,
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        filepath = Path(dialog.GetPath())
        dialog.Destroy()
        self._log(f'Loading crystal data from {filepath}.')
        self.app_frame.crystal = Crystal.from_json(filepath)
        wx.PostEvent(self.app_frame, CrystalChangedEvent())
        self._log('Crystal loaded.')

    def update_face_list(self, event):
        """
        Update the list of faces.
        """
        self.face_list.DeleteAllItems()
        for i, (hkl, d) in enumerate(zip(self.crystal.miller_indices, self.crystal.distances)):
            self.face_list.InsertItem(i, f'[{hkl[0]}, {hkl[1]}, {hkl[2]}]')
            self.face_list.SetItem(i, 1, f'{d:.2f}')
        event.Skip()

    def update_control_labels(self, event):
        """
        Update the text labels for rotation, origin position and refractive index.
        """
        self.lbl_rotation.SetLabel(label=f'Rotation: {np.rad2deg(to_numpy(self.crystal.rotation[2])):.1f}°')
        self.lbl_position.SetLabel(label=f'Origin: ({self.crystal.origin[0]:.1f}, {self.crystal.origin[1]:.1f})')
        self.lbl_ior.SetLabel(label=f'IOR: {self.crystal.material_ior.item():.1f}')
        event.Skip()

    def on_save_crystal(self, event):
        """
        Save the crystal details to a json file.
        """
        wildcard = 'Crystal|*.json'
        dialog = wx.FileDialog(self, message='Save Crystal', wildcard=wildcard,
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        filepath = dialog.GetPath()
        dialog.Destroy()
        self._log(f'Saving crystal data to {filepath}')
        self.crystal.to_json(filepath)
        self._log(f'Crystal saved to {filepath}.')

    def on_rotate(self, event):
        """
        Rotate the crystal.
        """
        if self.crystal is None:
            wx.MessageBox(message='You must load a crystal first.', caption='CrystalSizer3D',
                          style=wx.OK | wx.ICON_ERROR)
            return
        button = event.GetEventObject()
        label = button.GetLabel()
        assert label in ['↻', '↺']
        increment = self.STEP_ROTATION if label == '↺' else -self.STEP_ROTATION
        with torch.no_grad():
            self.crystal.rotation.data[2] += increment
        wx.PostEvent(self.app_frame, CrystalChangedEvent())

    def on_translate(self, event):
        """
        Translate the crystal.
        """
        if self.crystal is None:
            wx.MessageBox(message='You must load a crystal first.', caption='CrystalSizer3D',
                          style=wx.OK | wx.ICON_ERROR)
            return
        button = event.GetEventObject()
        label = button.GetLabel()
        assert label in ['↑', '↓', '←', '→']
        if label in ['←', '→']:
            idx = 0
        elif label in ['↑', '↓']:
            idx = 1
        if label in ['↑', '→']:
            increment = self.STEP_TRANSLATION
        elif label in ['↓', '←']:
            increment = -self.STEP_TRANSLATION
        with torch.no_grad():
            self.crystal.origin.data[idx] += increment
        wx.PostEvent(self.app_frame, CrystalChangedEvent())

    def on_distance_change(self, event):
        """
        Change one of the face distances of the crystal.
        """
        if self.crystal is None:
            wx.MessageBox(message='You must load a crystal first.', caption='CrystalSizer3D',
                          style=wx.OK | wx.ICON_ERROR)
            return
        idx = self.face_list.GetFocusedItem()
        if idx == -1:
            wx.MessageBox(message='You must select a face.', caption='CrystalSizer3D', style=wx.OK | wx.ICON_ERROR)
            return
        button = event.GetEventObject()
        label = button.GetLabel()
        assert label in ['Dis. +', 'Dis. -']
        increment = self.STEP_DISTANCE if label == 'Dis. +' else -self.STEP_DISTANCE
        with torch.no_grad():
            self.crystal.distances.data[idx] += increment
        wx.PostEvent(self.app_frame, CrystalChangedEvent())

    def on_scale_change(self, event):
        """
        Change the scale of the crystal.
        """
        if self.crystal is None:
            wx.MessageBox(message='You must load a crystal first.', caption='CrystalSizer3D',
                          style=wx.OK | wx.ICON_ERROR)
            return
        button = event.GetEventObject()
        label = button.GetLabel()
        assert label in ['Scl. +', 'Scl. -']
        sf = (1 + self.STEP_SCALE) if label == 'Scl. +' else (1 - self.STEP_SCALE)
        with torch.no_grad():
            self.crystal.distances.data *= sf
        wx.PostEvent(self.app_frame, CrystalChangedEvent())

    def on_ior_change(self, event):
        """
        Change the refractive index of the crystal.
        """
        if self.crystal is None:
            wx.MessageBox(message='You must load a crystal first.', caption='CrystalSizer3D',
                          style=wx.OK | wx.ICON_ERROR)
            return
        button = event.GetEventObject()
        label = button.GetLabel()
        assert label in ['+', '-']
        increment = self.STEP_IOR if label == '+' else -self.STEP_IOR
        with torch.no_grad():
            self.crystal.material_ior += increment
        wx.PostEvent(self.app_frame, CrystalChangedEvent())
