import math
from pathlib import Path

import cv2
import numpy as np
import wx

from app.components.app_panel import AppPanel
from app.components.utils import EVT_CRYSTAL_CHANGED, EVT_IMAGE_PATH_CHANGED, numpy_to_wx_image, wx_image_to_numpy
from crystalsizer3d.projector import Projector
from crystalsizer3d.util.utils import to_numpy


class ImagePanel(AppPanel):
    image: wx.Image = None
    image_combined: wx.Image = None
    projector: Projector = None
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

    def _init_components(self):
        """
        Initialise the image panel components.
        """
        # Image frame
        self.scrolled_window = wx.ScrolledWindow(self)
        self.scrolled_window.SetScrollRate(20, 20)
        self.scrolled_window.AlwaysShowScrollbars(True, True)
        self.scrolled_window.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)
        self.img_bitmap = wx.StaticBitmap(self.scrolled_window)
        self.img_bitmap.Bind(wx.EVT_LEFT_DOWN, self.on_click_image_L)
        self.img_bitmap.Bind(wx.EVT_RIGHT_DOWN, self.on_click_image_R)
        img_frame = wx.StaticBox(self, label='Image')
        img_sizer = wx.StaticBoxSizer(img_frame, wx.VERTICAL)
        img_sizer.Add(self.scrolled_window, 1, wx.EXPAND | wx.ALL, 10)
        scrolled_sizer = wx.BoxSizer(wx.VERTICAL)
        scrolled_sizer.Add(self.img_bitmap, 1, wx.EXPAND)
        self.scrolled_window.SetSizer(scrolled_sizer)

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
        main_sizer.Add(img_sizer, 7, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(controls_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(main_sizer)

    def _init_listeners(self):
        """
        Initialise the event listeners.
        """
        self.app_frame.Bind(EVT_IMAGE_PATH_CHANGED, self.load_image)
        self.app_frame.Bind(EVT_CRYSTAL_CHANGED, self.update_wireframe)

    def load_image(self, event):
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
        if self.projector is not None:
            self.projector.background_image = None
            self.update_wireframe(update_image=False)

        # Find the best zoom level for the image that fits it all in the frame
        img_width, img_height = image.GetSize()
        window_width, window_height = self.scrolled_window.GetClientSize()
        zoom_x = window_width / img_width
        zoom_y = window_height / img_height
        zoom = min(zoom_x, zoom_y)
        zoom = round(zoom * 10) / 10
        self.zoom = max(0.1, min(zoom, 10))

        # Update the image
        self.on_zoom_changed()
        self._log('Image loaded')
        event.Skip()

    def update_image(self):
        """
        Scale the image according to the current zoom level.
        """
        if self.image is None:
            return
        image = self.image_combined if self.image_combined is not None else self.image
        scaled_image = image.Scale(
            image.GetWidth() * self.zoom,
            image.GetHeight() * self.zoom,
            wx.IMAGE_QUALITY_HIGH
        )
        bitmap = wx.Bitmap(scaled_image)
        self.img_bitmap.SetBitmap(bitmap)

        # Set the virtual size of the scrolled window
        self.scrolled_window.SetVirtualSize(bitmap.GetSize())

        # Ensure the scrolled window shows scroll bars if necessary
        self.scrolled_window.SetScrollbars(1, 1, bitmap.GetWidth(), bitmap.GetHeight())

        # Refresh to update the display
        self.Refresh()

    def update_wireframe(self, event: wx.Event = None, update_image: bool = True):
        """
        Update the wireframe overlay.
        """
        if self.crystal is None:
            return
        self._log('Updating wireframe...')

        # Rebuild the crystal mesh
        self.crystal.build_mesh()

        # Initialise the projector
        w, h = self.image.GetSize()
        if self.projector is None or self.projector.image_size != (h, w):
            self._log('Initialising projector...')
            self.projector = Projector(self.crystal, image_size=(h, w))

        # Add overlay to image
        if self.image is not None and self.projector.background_image is None:
            self._log('Setting background image for projector...')
            bg_image = wx_image_to_numpy(self.image)
            self.projector.set_background(bg_image)

        # Convert the overlay to a wx.Image
        self._log('Projecting crystal wireframe...')
        img_overlay = to_numpy(self.projector.project() * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
        self._log('Updating image...')
        self.image_combined = numpy_to_wx_image(img_overlay)
        if update_image:
            self.update_image()
        self._log('Crystal wireframe updated.')
        if event is not None:
            event.Skip()

    def on_mouse_wheel(self, event):
        """
        Handle mouse wheel events.
        """
        if event.ShiftDown():
            # Horizontal scrolling
            scroll_units = -event.GetWheelRotation() / event.GetWheelDelta()
            self.scrolled_window.Scroll(self.scrolled_window.GetScrollPos(wx.HORIZONTAL) + scroll_units,
                                        self.scrolled_window.GetScrollPos(wx.VERTICAL))
        elif event.ControlDown():
            # Zoom in/out
            if event.GetWheelRotation() > 0:
                self.on_zoom_in()
            else:
                self.on_zoom_out()
        else:
            # Vertical scrolling (default behavior)
            event.Skip()

    def on_zoom_in(self, event=None):
        """
        Zoom in the image.
        """
        if self.zoom + self.STEP_ZOOM > 10:
            return
        self.zoom += self.STEP_ZOOM
        self.on_zoom_changed()

    def on_zoom_out(self, event=None):
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
        self.update_image()

    def on_click_image_L(self, event):
        self.clicL = event.GetPosition()
        self.update_measurer(None)

    def on_click_image_R(self, event):
        self.clicR = event.GetPosition()
        self.update_measurer(None)

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
