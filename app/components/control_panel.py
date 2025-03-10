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
    STEP_ROTATION = 0.005
    STEP_TRANSLATION = 0.01
    STEP_DISTANCE = 0.01
    STEP_SCALE = 0.02
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

        # Face list - with single selection only
        self.face_list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.face_list.InsertColumn(0, 'Miller Index')
        self.face_list.SetColumnWidth(col=0, width=int(self.face_list.GetSize()[0] / 2))
        self.face_list.InsertColumn(1, 'Distance')
        self.face_list.SetColumnWidth(col=1, width=int(self.face_list.GetSize()[0] / 2))
        self.face_list.SetMinSize((0, 400))
        self.face_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_face_list_select)

        # Panel for adjusting face normal distances
        self.txtctrl_dis = wx.TextCtrl(self, value='-')
        self.btn_distance_update = wx.Button(self, label='Update')
        self.btn_distance_update.Bind(wx.EVT_BUTTON, self.on_distance_change)
        txt_sizer = wx.BoxSizer(wx.HORIZONTAL)
        txt_sizer.Add(self.txtctrl_dis, 1, wx.EXPAND)
        txt_sizer.Add(self.btn_distance_update, 1, wx.EXPAND)

        self.btn_distance_up = wx.Button(self, label='Dis. +')
        self.btn_distance_up.Bind(wx.EVT_BUTTON, self.on_distance_change)
        self.btn_distance_down = wx.Button(self, label='Dis. -')
        self.btn_distance_down.Bind(wx.EVT_BUTTON, self.on_distance_change)

        d_sizer = wx.BoxSizer(wx.HORIZONTAL)
        d_sizer.Add(self.btn_distance_up, 1, wx.EXPAND)
        d_sizer.Add(self.btn_distance_down, 1, wx.EXPAND)

        # Panel for adjusting scale (normal distances * scale param)
        self.lbl_scale = wx.StaticText(self, label='Scale: ')
        self.btn_scale_up = wx.Button(self, label='Scl. +')
        self.btn_scale_up.Bind(wx.EVT_BUTTON, self.on_scale_change)
        self.btn_scale_down = wx.Button(self, label='Scl. -')
        self.btn_scale_down.Bind(wx.EVT_BUTTON, self.on_scale_change)
        self.btn_normalise = wx.Button(self, label='Normalise Dis & Scl')
        self.btn_normalise.Bind(wx.EVT_BUTTON, self.on_scale_change)

        s_sizer_btn = wx.BoxSizer(wx.HORIZONTAL)
        s_sizer_btn.Add(self.btn_scale_up, 1, wx.EXPAND)
        s_sizer_btn.Add(self.btn_scale_down, 1, wx.EXPAND)

        s_sizer = wx.BoxSizer(wx.VERTICAL)
        s_sizer.Add(self.lbl_scale, 1, wx.EXPAND | wx.ALL, 1)
        s_sizer.Add(s_sizer_btn, 1, wx.EXPAND | wx.ALL, 1)
        s_sizer.Add(self.btn_normalise, 1, wx.EXPAND | wx.ALL, 1)

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
        main_sizer.Add(txt_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
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
        dialog = wx.FileDialog(self, message='Load Image', wildcard='Images|*.jpg;*.png',
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        filepath = dialog.GetPath()
        try:
            image = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
            assert image.IsOk(), 'Invalid image file'
            self.app_frame.image_path = filepath
            wx.PostEvent(self.app_frame, ImagePathChangedEvent())
        except Exception as e:
            self._log('Loading image failed.')
            logger.error(f'Loading image failed: {e}')
            wx.MessageBox(message=str(e), caption='Error', style=wx.OK | wx.ICON_ERROR)
        dialog.Destroy()

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
        wx.PostEvent(self.app_frame, CrystalChangedEvent(build_mesh=False))
        self._log('Crystal loaded.')

    def on_face_list_select(self, event):
        idx = self.face_list.GetFirstSelected()
        dis = self.face_list.GetItem(idx, 1).GetText()
        self.txtctrl_dis.SetValue(dis)

    def update_face_list(self, event):
        """
        Update the list of faces.
        """
        event.Skip()
        idx = self.face_list.GetFirstSelected()
        self.face_list.DeleteAllItems()
        for i, (hkl, d) in enumerate(zip(self.crystal.miller_indices, self.crystal.distances)):
            self.face_list.InsertItem(i, f'[{hkl[0]}, {hkl[1]}, {hkl[2]}]')
            self.face_list.SetItem(i, 1, f'{d:.4f}')
        if idx != -1:
            self.face_list.Select(idx)

    def update_control_labels(self, event):
        """
        Update the text labels for rotation, origin position and refractive index.
        """
        event.Skip()
        self.lbl_rotation.SetLabel(label=f'Rotation: {np.rad2deg(to_numpy(self.crystal.rotation[2])):.1f}°')
        self.lbl_position.SetLabel(label=f'Origin: ({self.crystal.origin[0]:.2f}, {self.crystal.origin[1]:.2f})')
        self.lbl_ior.SetLabel(label=f'IOR: {self.crystal.material_ior.item():.2f}')
        self.lbl_scale.SetLabel(label=f'Scale: {self.crystal.scale.item():.2f}')

    def on_save_crystal(self, event: wx.CommandEvent):
        """
        Save the crystal details to a json file.
        """
        wildcard = 'Crystal|*.json'
        dialog = wx.FileDialog(self, message='Save Crystal', wildcard=wildcard,
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        filepath = Path(dialog.GetPath())
        dialog.Destroy()
        if filepath.suffix != '.json':
            filepath = filepath.with_suffix('.json')
        if filepath.exists():
            dlg = wx.MessageDialog(self, message='File already exists. Overwrite?', caption='CrystalSizer3D',
                                   style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
            if dlg.ShowModal() != wx.ID_YES:
                return
        self._log(f'Saving crystal data to {filepath}')
        self.crystal.to_json(Path(filepath), overwrite=True)
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
        idx = self.face_list.GetFirstSelected()
        if idx == -1:
            wx.MessageBox(message='You must select a face.', caption='CrystalSizer3D', style=wx.OK | wx.ICON_ERROR)
            return
        dis = self.face_list.GetItem(idx, 1).GetText()
        button = event.GetEventObject()
        label = button.GetLabel()
        assert label in ['Dis. +', 'Dis. -', 'Update']
        if label == 'Dis. +':
            increment = self.STEP_DISTANCE
        elif label == 'Dis. -':
            increment = -self.STEP_DISTANCE
        else:
            entry = self.txtctrl_dis.GetValue()
            increment = float(entry) - float(dis)
            # TODO: entry sanitization to avoid crashing.
            # How to determine whether the entry is a valid distance or not?
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
        assert label in ['Scl. +', 'Scl. -', 'Normalise Dis & Scl']
        # TODO: replace strings with constants at top of class (all btn labels & their evt)
        if label == 'Scl. +':
            sf = self.STEP_SCALE
        elif label == 'Scl. -':
            sf = - self.STEP_SCALE
        else:
            # Update distance here and adjusting scale
            sd = 1 / np.max(self.crystal.distances.data.tolist())
        with torch.no_grad():
            if label == 'Normalise Dis & Scl':
                self.crystal.distances.data *= sd
                self.crystal.scale /= sd
            else:
                self.crystal.scale += sf
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
