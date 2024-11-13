import math
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import wx

from app import DENOISED_IMAGE_PATH, SCENE_IMAGE_PATH
from app.components.anchor_manager import AnchorManager
from app.components.app_panel import AppPanel
from app.components.utils import EVT_CRYSTAL_MESH_CHANGED, EVT_DENOISED_IMAGE_CHANGED, EVT_IMAGE_PATH_CHANGED, \
    EVT_SCENE_IMAGE_CHANGED, ImagePathChangedEvent, numpy_to_wx_image
from crystalsizer3d.util.utils import to_numpy


class ImagePanel(AppPanel):
    image_windows: Dict[str, wx.ScrolledWindow] = {
        'image': None,
        'denoised': None,
        'scene': None
    }
    image_containers: Dict[str, wx.StaticBitmap] = {
        'image': None,
        'denoised': None,
        'scene': None
    }
    images: Dict[str, Optional[wx.Image]] = {
        'image': None,
        'denoised': None,
        'scene': None,
        'wireframe': None,
        'anchors': None
    }
    bitmaps: Dict[str, wx.Bitmap] = {
        'image': None,
        'denoised': None,
        'scene': None,
        'wireframe': None,
        'anchors': None
    }
    active_window: str = 'image'
    scene_image_needs_loading: bool = False
    images_updating: bool = False
    scroll_x: int = 0
    scroll_y: int = 0
    is_zooming: bool = False
    STEP_ZOOM = 0.1

    # Manual constraints management

    highlight_vertex_idx: Optional[int] = None
    selected_vertex_idx: Optional[int] = None

    def __init__(self, app_frame: 'AppFrame'):
        # Magnification - zoom in/out the image
        self.zoom = 1.0

        # Clicked points - for measurement
        self.clicL = [0, 0]
        self.clicR = [0, 0]

        # Initialise panel
        super().__init__(app_frame)

        # Initialise the AnchorManager
        self.anchor_manager = AnchorManager(self)

    @property
    def image_path(self) -> Path:
        return self.app_frame.image_path

    def _init_components(self):
        """
        Initialise the image panel components.
        """

        # Image tabs
        self.image_tabs = wx.Notebook(self)
        labels = {
            'image': 'Image',
            'denoised': 'Denoised',
            'scene': 'Scene'
        }
        for k, lbl in labels.items():
            window = wx.ScrolledWindow(self.image_tabs)
            window.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            window.SetScrollRate(20, 20)
            window.AlwaysShowScrollbars(True, True)
            window.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)
            container = wx.StaticBitmap(window)
            img_frame = wx.StaticBox(self, label=lbl)
            img_sizer = wx.StaticBoxSizer(img_frame, wx.VERTICAL)
            img_sizer.Add(window, 1, wx.EXPAND | wx.ALL, 10)
            scrolled_sizer = wx.BoxSizer(wx.VERTICAL)
            scrolled_sizer.Add(container, 1, wx.EXPAND)
            window.SetSizer(scrolled_sizer)
            self.image_windows[k] = window
            self.image_containers[k] = container
            self.image_tabs.AddPage(window, lbl)

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
        self.btn_zoom_in = wx.Button(self, label='+')
        self.btn_zoom_in.Bind(wx.EVT_BUTTON, self.on_zoom_in)
        self.btn_zoom_out = wx.Button(self, label='-')
        self.btn_zoom_out.Bind(wx.EVT_BUTTON, self.on_zoom_out)
        zoom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        zoom_sizer.Add(self.lbl_zoom, 2, wx.EXPAND | wx.ALL, 2)
        zoom_sizer.Add(self.btn_zoom_in, 1, wx.EXPAND | wx.ALL, 2)
        zoom_sizer.Add(self.btn_zoom_out, 1, wx.EXPAND | wx.ALL, 2)

        # Wireframe checkbox
        self.ckbx_wireframe = wx.CheckBox(self, label='Show wireframe')
        self.ckbx_wireframe.SetValue(True)
        self.ckbx_wireframe.Bind(wx.EVT_CHECKBOX, self.on_wireframe_ckbx)

        controls_sizer_r = wx.StaticBoxSizer(wx.VERTICAL, parent=self)
        controls_sizer_r.Add(zoom_sizer, 1, wx.EXPAND | wx.ALL, 1)
        controls_sizer_r.Add(self.ckbx_wireframe, 1, wx.EXPAND | wx.ALL, 1)

        # Controls sizer
        controls_sizer = wx.BoxSizer(wx.HORIZONTAL)
        controls_sizer.Add(measurement_sizer, 1, wx.EXPAND | wx.ALL, 2)
        controls_sizer.Add(controls_sizer_r, 1, wx.EXPAND | wx.ALL, 2)

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.image_tabs, 7, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(controls_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(main_sizer)

    def _init_listeners(self):
        """
        Initialise the event listeners.
        """
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_image_tab_changed)
        self.app_frame.Bind(EVT_IMAGE_PATH_CHANGED, self.load_image)
        self.app_frame.Bind(EVT_DENOISED_IMAGE_CHANGED, self.load_denoised_image)
        self.app_frame.Bind(EVT_SCENE_IMAGE_CHANGED, self.load_scene_image)
        self.app_frame.Bind(EVT_CRYSTAL_MESH_CHANGED, self.update_wireframe)
        self.image_tabs.Bind(wx.EVT_SIZE, self.find_best_zoom)

    def on_image_tab_changed(self, event: wx.Event):
        """
        Handle image tab change events.
        """
        self.active_window = ['image', 'denoised', 'scene'][event.GetSelection()]
        self.update_images(quiet=True)
        window = self.image_windows[self.active_window]
        window.Scroll(round(self.scroll_x), round(self.scroll_y))

    def find_best_zoom(self, event: wx.Event = None):
        """
        Find the best zoom level for the image.
        """
        if event is not None:
            event.Skip()
        image = self.images['image']
        if image is None:
            return
        img_width, img_height = self.images['image'].GetSize()
        window_width, window_height = self.image_tabs.GetClientSize()
        zoom_x = window_width / img_width
        zoom_y = window_height / img_height
        zoom = min(zoom_x, zoom_y)
        zoom = round(zoom * 10) / 10
        new_zoom = max(0.1, min(zoom, 10))
        if new_zoom != self.zoom:
            self.zoom = new_zoom
            self.on_zoom_changed()

    def load_image(self, event: ImagePathChangedEvent):
        """
        Load an image from disk.
        """
        self._log(f'Loading image from {self.image_path}...')
        image = wx.Image(str(self.image_path), wx.BITMAP_TYPE_ANY)
        if not image.IsOk():
            wx.MessageBox(message='Invalid image file', caption='Error', style=wx.OK | wx.ICON_ERROR),
            self._log('Failed to load image: Invalid image file.')
            return
        self.images['image'] = image
        initial_load = event.initial_load if hasattr(event, 'initial_load') else False
        if not initial_load:
            self.images['denoised'] = None
            self.images['scene'] = None
            if DENOISED_IMAGE_PATH.exists():
                DENOISED_IMAGE_PATH.unlink()
            if SCENE_IMAGE_PATH.exists():
                SCENE_IMAGE_PATH.unlink()
            self.update_wireframe(update_images=False)
        wx.CallLater(100, self.find_best_zoom)
        self._log('Image loaded.')
        event.Skip()

    def update_images(self, quiet: bool = False):
        """
        Scale the images according to the current zoom level.
        """
        if self.images['image'] is None:
            return
        if self.images_updating:
            return
        self.images_updating = True
        if not quiet:
            self._log('Updating images...')
        base_image = self.images['image']

        # Scale the wireframe overlay for use on all images
        if self.images['wireframe'] is not None:
            wireframe = self.images['wireframe']
            sf = base_image.GetHeight() / wireframe.GetHeight()
            scaled_wireframe_width = round(wireframe.GetWidth() * sf * self.zoom)
            scaled_wireframe_height = round(wireframe.GetHeight() * sf * self.zoom)
            scaled_wireframe = wireframe.Scale(
                scaled_wireframe_width,
                scaled_wireframe_height,
                wx.IMAGE_QUALITY_HIGH
            )
            self.bitmaps['wireframe'] = wx.Bitmap(scaled_wireframe)

            # Scale the anchors overlay for use on all images
            if self.images['anchors'] is not None:
                anchors = self.images['anchors']
                scaled_anchors = anchors.Scale(
                    scaled_wireframe_width,
                    scaled_wireframe_height,
                    wx.IMAGE_QUALITY_HIGH
                )
                self.bitmaps['anchors'] = wx.Bitmap(scaled_anchors)

        # Load the scene image if it needs loading
        if self.active_window == 'scene' and self.scene_image_needs_loading:
            self.load_scene_image(update_images=False)
            self.scene_image_needs_loading = False

        # Get the active image and its containers
        image = self.images[self.active_window]
        window = self.image_windows[self.active_window]
        container = self.image_containers[self.active_window]
        window.Freeze()

        # If there is no image to be shown, clear the container and return
        if image is None or not image.IsOk():
            container.SetBitmap(wx.NullBitmap)
            self.Refresh()
            if not quiet:
                self._log('Images updated.')
            return

        # Scale the image to match the height of the main image
        sf = base_image.GetHeight() / image.GetHeight()
        scaled_image = image.Scale(
            round(image.GetWidth() * sf * self.zoom),
            round(image.GetHeight() * sf * self.zoom),
            wx.IMAGE_QUALITY_HIGH
        )
        bitmap = wx.Bitmap(scaled_image)
        self.bitmaps[self.active_window] = bitmap

        # Draw the wireframe and anchors overlays on the image
        if self.images['wireframe'] is not None and self.ckbx_wireframe.IsChecked():
            image_width, image_height = bitmap.GetSize()
            wireframe_x = (image_width - scaled_wireframe_width) // 2
            wireframe_y = (image_height - scaled_wireframe_height) // 2
            mem_dc = wx.MemoryDC()
            mem_dc.SelectObject(bitmap)
            mem_dc.DrawBitmap(self.bitmaps['wireframe'], wireframe_x, wireframe_y, True)
            if self.images['anchors'] is not None:
                mem_dc.DrawBitmap(self.bitmaps['anchors'], wireframe_x, wireframe_y, True)
            mem_dc.SelectObject(wx.NullBitmap)

        # Update the image container
        container.SetBitmap(bitmap)

        # Restore the scroll position
        window.Layout()
        if window.GetVirtualSize() != bitmap.GetSize():
            window.SetVirtualSize(bitmap.GetSize())
        window.SetScrollbars(1, 1, bitmap.GetWidth(), bitmap.GetHeight())
        window.SetScrollRate(int(20 * self.zoom), int(20 * self.zoom))
        window.Scroll(round(self.scroll_x), round(self.scroll_y))
        window.Thaw()
        window.Refresh()

        if not quiet:
            self._log('Images updated.')
        self.images_updating = False

    def update_wireframe(self, event: wx.Event = None, update_images: bool = True):
        """
        Update the wireframe overlay.
        """
        if self.crystal is None:
            return
        if event is not None:
            event.Skip()
        if id(self.projector.crystal) != id(self.crystal):
            self.projector.crystal = self.crystal
        self._log('Updating wireframe...')

        # Project and convert to a wx.Image
        self._log('Projecting crystal wireframe...')
        wireframe = to_numpy(self.projector.project() * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
        self.images['wireframe'] = numpy_to_wx_image(wireframe)

        # Scale the wireframe bitmap
        base_image = self.images['image']
        wireframe = self.images['wireframe']
        sf = base_image.GetHeight() / wireframe.GetHeight()
        scaled_wireframe_width = round(wireframe.GetWidth() * sf * self.zoom)
        scaled_wireframe_height = round(wireframe.GetHeight() * sf * self.zoom)
        scaled_wireframe = wireframe.Scale(
            scaled_wireframe_width,
            scaled_wireframe_height,
            wx.IMAGE_QUALITY_HIGH
        )
        self.bitmaps['wireframe'] = wx.Bitmap(scaled_wireframe)

        # Update the anchors overlay
        if len(self.anchor_manager.anchors) > 0:
            self.anchor_manager.update_overlay(update_images=False)

        # Update the images with the new wireframe and anchors overlays
        if update_images:
            self.update_images()
        self._log('Crystal wireframe updated.')

    def load_denoised_image(self, event: wx.Event = None, update_images: bool = True):
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
        if not image.IsSameAs(wx.Image()) and not image.IsOk():
            wx.MessageBox(message='Invalid image file', caption='Error', style=wx.OK | wx.ICON_ERROR),
            self._log('Failed to load denoised image: Invalid image file.')
            return
        self._log('Loading denoised image...')
        self.images['denoised'] = image
        if update_images:
            self.update_images()
        image.SaveFile(str(DENOISED_IMAGE_PATH), wx.BITMAP_TYPE_PNG)
        self._log('Denoised image loaded.')

    def load_scene_image(self, event: wx.Event = None, update_images: bool = True):
        """
        Load the scene image.
        """
        if event is not None:
            event.Skip()
        self.scene_image_needs_loading = True
        if self.active_window != 'scene':
            return
        if self.app_frame.refiner is None:
            if SCENE_IMAGE_PATH.exists():
                image = wx.Image(str(SCENE_IMAGE_PATH), wx.BITMAP_TYPE_ANY)
            else:
                image = wx.Image()
        else:
            X_pred = self.app_frame.refiner.X_pred
            if X_pred is not None:
                image = numpy_to_wx_image(to_numpy(X_pred * 255).astype(np.uint8))
            else:
                image = wx.Image()
        if not image.IsSameAs(wx.Image()) and not image.IsOk():
            wx.MessageBox(message='Invalid image file', caption='Error', style=wx.OK | wx.ICON_ERROR),
            self._log('Failed to load scene image: Invalid image file.')
            return
        self._log('Loading scene image...')
        self.images['scene'] = image
        self.scene_image_needs_loading = False
        if event is not None and hasattr(event, 'update_images'):
            update_images = event.update_images
        if update_images:
            self.update_images()
        if not image.IsSameAs(wx.Image()):
            image.SaveFile(str(SCENE_IMAGE_PATH), wx.BITMAP_TYPE_PNG)
        self._log('Scene image loaded.')

    def on_mouse_wheel(self, event: wx.MouseEvent):
        """
        Handle mouse wheel events.
        """
        # Zoom in/out
        if event.ControlDown():
            if event.GetWheelRotation() > 0:
                self.on_zoom_in()
            else:
                self.on_zoom_out()

        # Scrolling
        else:
            scroll_units = -event.GetWheelRotation() / event.GetWheelDelta()
            window = event.GetEventObject()
            if event.ShiftDown():
                self.scroll_x = window.GetScrollPos(wx.HORIZONTAL) + scroll_units
                self.scroll_y = window.GetScrollPos(wx.VERTICAL)
            else:
                self.scroll_x = window.GetScrollPos(wx.HORIZONTAL)
                self.scroll_y = window.GetScrollPos(wx.VERTICAL) + scroll_units
            window.Scroll(round(self.scroll_x), round(self.scroll_y))

    def on_zoom_in(self, event: wx.Event = None):
        """
        Zoom in the image.
        """
        if self.is_zooming:
            return
        if self.zoom + self.STEP_ZOOM > 10:
            return
        self.is_zooming = True
        self.zoom += self.STEP_ZOOM
        self.on_zoom_changed()

    def on_zoom_out(self, event: wx.Event = None):
        """
        Zoom out the image.
        """
        if self.is_zooming:
            return
        if self.zoom - self.STEP_ZOOM < 0.1:
            return
        self.is_zooming = True
        self.zoom -= self.STEP_ZOOM
        self.on_zoom_changed()

    def on_zoom_changed(self):
        """
        Update the image when the zoom level changes.
        """
        self.lbl_zoom.SetLabel(label=f'Zoom: {self.zoom:.1f}x')
        self.update_images()
        self.anchor_manager.update_overlay(update_images=True)

        def after_zoom_finished():
            self.is_zooming = False

        # Delay setting the flag to debounce zoom events
        wx.CallLater(100, after_zoom_finished)

    def on_wireframe_ckbx(self, event):
        """
        Update image after changing status of checkbox 'show wireframe'
        """
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
