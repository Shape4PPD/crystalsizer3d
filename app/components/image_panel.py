import math
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import wx
from ruamel.yaml import YAML

from app import APP_ASSETS_PATH, DENOISED_IMAGE_PATH, SCENE_ARGS_PATH, SCENE_IMAGE_PATH
from app.components.app_panel import AppPanel
from app.components.utils import EVT_CRYSTAL_CHANGED, EVT_DENOISED_IMAGE_CHANGED, EVT_IMAGE_PATH_CHANGED, \
    EVT_SCENE_IMAGE_CHANGED, numpy_to_wx_image
from crystalsizer3d.projector import Projector
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.util.utils import to_numpy


class ImagePanel(AppPanel):
    image: wx.Image = None
    image_denoised: Optional[wx.Image] = None
    image_scene: Optional[wx.Image] = None
    wireframe: Optional[wx.Image] = None
    scene: Optional[Scene] = None
    projector: Optional[Projector] = None
    STEP_ZOOM = 0.1

    def __init__(self, app_frame: 'AppFrame'):
        # Magnification - zoom in/out the image
        self.zoom = 1.0

        # Clicked points - for measurement
        self.clicL = [0, 0]
        self.clicR = [0, 0]

        # Initialise panel
        super().__init__(app_frame)

    @property
    def image_path(self) -> Path:
        return self.app_frame.image_path

    def _init_scene(self):
        """
        Initialise the scene for rendering.
        """
        self._log('Initialising scene...')
        yaml = YAML()
        yaml.preserve_quotes = True

        # Copy over default scene args if they don't exist
        if not SCENE_ARGS_PATH.exists():
            shutil.copy(APP_ASSETS_PATH / 'default_scene_args.yml', SCENE_ARGS_PATH)

        # Load the scene args from file
        with open(SCENE_ARGS_PATH, 'r') as f:
            args = yaml.load(f)

        # Instantiate the scene
        try:
            self.scene = Scene(
                crystal=self.crystal,
                res=min(self.image.GetHeight(), self.image.GetWidth()),
                **args
            )
        except Exception as e:
            wx.MessageBox(message=str(e), caption='Error initialising scene',
                          style=wx.OK | wx.ICON_ERROR)
            self._log(f'Error initialising scene.')
            return

    def _init_components(self):
        """
        Initialise the image panel components.
        """

        # Tabs
        image_tabs = wx.Notebook(self)

        # Image tab
        self.image_window = wx.ScrolledWindow(image_tabs)
        self.image_window.SetScrollRate(20, 20)
        self.image_window.AlwaysShowScrollbars(True, True)
        self.img_bitmap = wx.StaticBitmap(self.image_window)
        self.img_bitmap.Bind(wx.EVT_LEFT_DOWN, self.on_click_image_L)
        self.img_bitmap.Bind(wx.EVT_RIGHT_DOWN, self.on_click_image_R)
        img_frame = wx.StaticBox(self, label='Image')
        img_sizer = wx.StaticBoxSizer(img_frame, wx.VERTICAL)
        img_sizer.Add(self.image_window, 1, wx.EXPAND | wx.ALL, 10)
        scrolled_sizer = wx.BoxSizer(wx.VERTICAL)
        scrolled_sizer.Add(self.img_bitmap, 1, wx.EXPAND)
        self.image_window.SetSizer(scrolled_sizer)

        # Denoised image tab
        self.image_window_dn = wx.ScrolledWindow(image_tabs)
        self.image_window_dn.SetScrollRate(20, 20)
        self.image_window_dn.AlwaysShowScrollbars(True, True)
        self.img_bitmap_dn = wx.StaticBitmap(self.image_window_dn)
        img_frame_dn = wx.StaticBox(self, label='Denoised Image')
        img_sizer_dn = wx.StaticBoxSizer(img_frame_dn, wx.VERTICAL)
        img_sizer_dn.Add(self.image_window_dn, 1, wx.EXPAND | wx.ALL, 10)
        scrolled_sizer_dn = wx.BoxSizer(wx.VERTICAL)
        scrolled_sizer_dn.Add(self.img_bitmap_dn, 1, wx.EXPAND)
        self.image_window_dn.SetSizer(scrolled_sizer_dn)

        # Rendered scene tab
        self.scene_window = wx.ScrolledWindow(image_tabs)
        self.scene_window.SetScrollRate(20, 20)
        self.scene_window.AlwaysShowScrollbars(True, True)
        self.img_bitmap_scene = wx.StaticBitmap(self.scene_window)
        img_frame_scene = wx.StaticBox(self, label='Rendered Scene')
        img_sizer_scene = wx.StaticBoxSizer(img_frame_scene, wx.VERTICAL)
        img_sizer_scene.Add(self.scene_window, 1, wx.EXPAND | wx.ALL, 10)
        scrolled_sizer_scene = wx.BoxSizer(wx.VERTICAL)
        scrolled_sizer_scene.Add(self.img_bitmap_scene, 1, wx.EXPAND)
        self.scene_window.SetSizer(scrolled_sizer_scene)

        # Add tabs
        image_tabs.AddPage(self.image_window, 'Image')
        image_tabs.AddPage(self.image_window_dn, 'Denoised')
        image_tabs.AddPage(self.scene_window, 'Scene')
        for window in [self.image_window, self.image_window_dn, self.scene_window]:
            window.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)
        self.image_tabs = image_tabs

        # Measurement controls
        self.lbl_start_pt = wx.StaticText(self, label='Start Pt: (0, 0)')
        self.lbl_end_pt = wx.StaticText(self, label='End Pt: (0, 0)')
        self.lbl_measurement = wx.StaticText(self, label='Distance: N/A')
        self.btn_clear_measurement = wx.Button(self, label='CLR Measurements')
        self.btn_clear_measurement.Bind(wx.EVT_BUTTON, self.on_clear_measure)
        m_sizer_l = wx.BoxSizer(wx.VERTICAL)
        m_sizer_l.Add(self.lbl_start_pt, 1, wx.ALIGN_LEFT | wx.ALL, 2)
        m_sizer_l.Add(self.lbl_end_pt, 1, wx.ALIGN_LEFT | wx.ALL, 2)
        m_sizer_r = wx.BoxSizer(wx.VERTICAL)
        m_sizer_r.Add(self.lbl_measurement, 1, wx.ALIGN_LEFT | wx.ALL, 2)
        m_sizer_r.Add(self.btn_clear_measurement, 1, wx.EXPAND | wx.ALL, 2)
        measurement_sizer = wx.StaticBoxSizer(wx.HORIZONTAL, parent=self, label='Measurer')
        measurement_sizer.Add(m_sizer_l, 1, wx.ALL, 1)
        measurement_sizer.Add(m_sizer_r, 1, wx.ALL, 1)

        # Zoom controls
        self.lbl_zoom = wx.StaticText(self, label=f'Zoom: {self.zoom:.1f}x')
        self.btn_zoom_in = wx.Button(self, label='Zoom in')
        self.btn_zoom_in.Bind(wx.EVT_BUTTON, self.on_zoom_in)
        self.btn_zoom_out = wx.Button(self, label='Zoom out')
        self.btn_zoom_out.Bind(wx.EVT_BUTTON, self.on_zoom_out)
        zoom_sizer_t = wx.BoxSizer(wx.HORIZONTAL)
        zoom_sizer_t.Add(self.lbl_zoom, wx.EXPAND | wx.ALL, 5)
        zoom_sizer_b = wx.BoxSizer(wx.HORIZONTAL)
        zoom_sizer_b.Add(self.btn_zoom_in, 1, wx.EXPAND | wx.ALL, 2)
        zoom_sizer_b.Add(self.btn_zoom_out, 1, wx.EXPAND | wx.ALL, 2)
        zoom_sizer = wx.StaticBoxSizer(wx.VERTICAL, parent=self)
        zoom_sizer.Add(zoom_sizer_t, 1, wx.EXPAND | wx.ALL, 1)
        zoom_sizer.Add(zoom_sizer_b, 1, wx.EXPAND | wx.ALL, 1)

        # Controls sizer
        controls_sizer = wx.BoxSizer(wx.HORIZONTAL)
        controls_sizer.Add(measurement_sizer, 1, wx.EXPAND | wx.ALL, 2)
        controls_sizer.Add(zoom_sizer, 1, wx.EXPAND | wx.ALL, 2)

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(image_tabs, 7, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(controls_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(main_sizer)

    def _init_listeners(self):
        """
        Initialise the event listeners.
        """
        self.app_frame.Bind(EVT_IMAGE_PATH_CHANGED, self.load_image)
        self.app_frame.Bind(EVT_DENOISED_IMAGE_CHANGED, self.load_denoised_image)
        self.app_frame.Bind(EVT_SCENE_IMAGE_CHANGED, self.load_scene_image)
        self.app_frame.Bind(EVT_CRYSTAL_CHANGED, self.update_wireframe)

    def load_image(self, event: wx.Event):
        """
        Load an image from disk.
        """
        self._log(f'Loading image from {self.image_path}...')
        image = wx.Image(str(self.image_path), wx.BITMAP_TYPE_ANY)
        if not image.IsOk():
            wx.MessageBox(message='Invalid image file', caption='Error', style=wx.OK | wx.ICON_ERROR),
            self._log('Failed to load image: Invalid image file.')
            return
        self.image = image
        self.image_denoised = None
        self.image_scene = None
        if DENOISED_IMAGE_PATH.exists():
            DENOISED_IMAGE_PATH.unlink()
        if SCENE_IMAGE_PATH.exists():
            SCENE_IMAGE_PATH.unlink()
        if self.scene is not None:
            self.scene.res = min(image.GetHeight(), image.GetWidth())
        if self.projector is not None:
            self.projector.background_image = None
            self.update_wireframe(update_images=False)

        # Find the best zoom level for the image that fits it all in the frame
        def find_best_zoom():
            img_width, img_height = image.GetSize()
            window_width, window_height = self.image_tabs.GetClientSize()
            zoom_x = window_width / img_width
            zoom_y = window_height / img_height
            zoom = min(zoom_x, zoom_y)
            zoom = round(zoom * 10) / 10
            self.zoom = max(0.1, min(zoom, 10))

            # Update the images
            self.on_zoom_changed()
            self._log('Image loaded.')

        wx.CallLater(0, find_best_zoom)
        event.Skip()

    def update_images(self):
        """
        Scale the images according to the current zoom level.
        """
        if self.image is None:
            return

        images = [self.image, self.image_denoised, self.image_scene]
        bitmaps = [self.img_bitmap, self.img_bitmap_dn, self.img_bitmap_scene]
        windows = [self.image_window, self.image_window_dn, self.scene_window]

        for image, bitmap, window in zip(images, bitmaps, windows):
            if image is None:
                bitmap.SetBitmap(wx.NullBitmap)
                continue

            sf = self.image.GetHeight() / image.GetHeight()
            scaled_image = image.Scale(
                round(image.GetWidth() * sf * self.zoom),
                round(image.GetHeight() * sf * self.zoom),
                wx.IMAGE_QUALITY_HIGH
            )
            bitmap_img = wx.Bitmap(scaled_image)

            if self.wireframe is not None:
                sf = self.image.GetHeight() / self.wireframe.GetHeight()
                scaled_wireframe = self.wireframe.Scale(
                    round(self.wireframe.GetWidth() * sf * self.zoom),
                    round(self.wireframe.GetHeight() * sf * self.zoom),
                    wx.IMAGE_QUALITY_HIGH
                )
                wireframe_bitmap = wx.Bitmap(scaled_wireframe)
                mem_dc = wx.MemoryDC()
                mem_dc.SelectObject(bitmap_img)
                mem_dc.DrawBitmap(wireframe_bitmap, 0, 0, True)
                mem_dc.SelectObject(wx.NullBitmap)

            bitmap.SetBitmap(bitmap_img)
            window.SetVirtualSize(bitmap_img.GetSize())
            window.SetScrollbars(1, 1, bitmap_img.GetWidth(), bitmap_img.GetHeight())
        self.Refresh()

    def update_wireframe(self, event: wx.Event = None, update_images: bool = True):
        """
        Update the wireframe overlay.
        """
        if self.crystal is None:
            return
        self._log('Updating wireframe...')

        # Rebuild the crystal mesh
        with torch.no_grad():
            self.crystal.build_mesh()

        # Initialise the projector
        w, h = self.image.GetSize()
        if self.projector is None or self.projector.image_size != (h, w):
            self._init_scene()
            self._log('Initialising projector...')
            self.projector = Projector(
                crystal=self.crystal,
                image_size=(h, w),
                zoom=orthographic_scale_factor(self.scene)
            )

        # Check that the crystal is the same
        if id(self.crystal) != id(self.projector.crystal):
            self.projector.crystal = self.crystal

        # Convert the overlay to a wx.Image
        self._log('Projecting crystal wireframe...')
        wireframe = to_numpy(self.projector.project() * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)

        # Add alpha channel to wireframe
        alpha = np.zeros((wireframe.shape[0], wireframe.shape[1], 1), dtype=np.uint8)
        alpha[wireframe.sum(axis=2) == 0] = 0
        alpha[wireframe.sum(axis=2) != 0] = 255
        wireframe = np.concatenate([wireframe, alpha], axis=2)

        self.wireframe = numpy_to_wx_image(wireframe)
        if update_images:
            self.update_images()
        self._log('Crystal wireframe updated.')
        if event is not None:
            event.Skip()

    def load_denoised_image(self, event: wx.Event = None):
        """
        Load the denoised image.
        """
        if event is not None:
            event.Skip()
        if self.app_frame.refiner is None:
            if DENOISED_IMAGE_PATH.exists():
                image = wx.Image(str(DENOISED_IMAGE_PATH), wx.BITMAP_TYPE_ANY)
            else:
                image = wx.Image()
        else:
            image = numpy_to_wx_image(to_numpy(self.app_frame.refiner.X_target_denoised * 255).astype(np.uint8))
        self._log('Loading denoised image...')
        if not image.IsOk():
            wx.MessageBox(message='Invalid image file', caption='Error', style=wx.OK | wx.ICON_ERROR),
            self._log('Failed to load denoised image: Invalid image file.')
            return
        self.image_denoised = image
        self.update_images()
        image.SaveFile(str(DENOISED_IMAGE_PATH), wx.BITMAP_TYPE_PNG)
        self._log('Denoised image loaded.')

    def load_scene_image(self, event: wx.Event = None):
        """
        Load the scene image.
        """
        if event is not None:
            event.Skip()
        if self.app_frame.refiner is None:
            if SCENE_IMAGE_PATH.exists():
                image = wx.Image(str(SCENE_IMAGE_PATH), wx.BITMAP_TYPE_ANY)
            else:
                image = wx.Image()
        else:
            image = numpy_to_wx_image(to_numpy(self.app_frame.refiner.X_pred * 255).astype(np.uint8))
        self._log('Loading scene image...')
        if not image.IsOk():
            wx.MessageBox(message='Invalid image file', caption='Error', style=wx.OK | wx.ICON_ERROR),
            self._log('Failed to load scene image: Invalid image file.')
            return
        self.image_scene = image
        self.update_images()
        image.SaveFile(str(SCENE_IMAGE_PATH), wx.BITMAP_TYPE_PNG)
        self._log('Scene image loaded.')

    def on_mouse_wheel(self, event: wx.MouseEvent):
        """
        Handle mouse wheel events.
        """
        if event.ShiftDown():
            # Horizontal scrolling
            scroll_units = -event.GetWheelRotation() / event.GetWheelDelta()
            window = event.GetEventObject()
            window.Scroll(window.GetScrollPos(wx.HORIZONTAL) + scroll_units,
                          window.GetScrollPos(wx.VERTICAL))
        elif event.ControlDown():
            # Zoom in/out
            if event.GetWheelRotation() > 0:
                self.on_zoom_in()
            else:
                self.on_zoom_out()
        else:
            # Vertical scrolling (default behavior)
            event.Skip()

    def on_zoom_in(self, event: wx.Event = None):
        """
        Zoom in the image.
        """
        if self.zoom + self.STEP_ZOOM > 10:
            return
        self.zoom += self.STEP_ZOOM
        self.on_zoom_changed()

    def on_zoom_out(self, event: wx.Event = None):
        """
        Zoom out the image.
        """
        if self.zoom - self.STEP_ZOOM < 0.1:
            return
        self.zoom -= self.STEP_ZOOM
        self.on_zoom_changed()

    def on_zoom_changed(self):
        """
        Update the image when the zoom level changes.
        """
        self.lbl_zoom.SetLabel(label=f'Zoom: {self.zoom:.1f}x')
        self.update_images()

    def on_click_image_L(self, event):
        self.clicL = event.GetPosition()
        # self.update_measurer(None)

    def on_click_image_R(self, event):
        self.clicR = event.GetPosition()
        # self.update_measurer(None)

    def update_measurer(self, event):
        self.lbl_start_pt.SetLabel(label='Start Pt: ' + str(self.clicL[0]) + ', ' + str(self.clicL[1]))
        self.lbl_end_pt.SetLabel(label='End Pt: ' + str(self.clicR[0]) + ', ' + str(self.clicR[1]))

        # Calculate distance between two points
        dis = math.sqrt(((self.clicR[0] - self.clicL[0])**2) + ((self.clicR[1] - self.clicL[1])**2))
        self.lbl_measurement.SetLabel(label='Distance: ' + str(int(dis)) + ' px')

        # Try: draw a point on the place clicked
        image = wx.ImageFromBitmap(self.img_bitmap.GetBitmap())
        buf = image.GetDataBuffer()  # use img.GetAlphaBuffer() for alpha data
        imgArr = np.frombuffer(buf, dtype='uint8')
        imgArr = np.reshape(imgArr, (self.image_size[1], self.image_size[0], 3))

        if self.clicL != [0, 0] and self.clicR != [0, 0]:
            cv2.line(imgArr, [self.clicL[0], self.clicL[1]],
                     [self.clicR[0], self.clicR[1]], [255, 255, 255], 2)
        if self.clicL != [0, 0]:
            cv2.circle(imgArr, [self.clicL[0], self.clicL[1]], 5, [255, 0, 0], -1)
        if self.clicR != [0, 0]:
            cv2.circle(imgArr, [self.clicR[0], self.clicR[1]], 5, [0, 0, 255], -1)

        wxImg = wx.Image(imgArr.shape[1], imgArr.shape[0])
        wxImg.SetData(imgArr.tobytes())
        self.img_bitmap.SetBitmap(wxImg.ConvertToBitmap())

    def on_clear_measure(self, event=None):
        self.update_wireframe()
        self.clicL = [0, 0]
        self.clicR = [0, 0]
        self.lbl_start_pt.SetLabel(label='Start Pt: ' + str(self.clicL[0]) + ', ' + str(self.clicL[1]))
        self.lbl_end_pt.SetLabel(label='End Pt: ' + str(self.clicR[0]) + ', ' + str(self.clicR[1]))
        self.lbl_measurement.SetLabel(label='Distance: N/A')
