import math
from collections import OrderedDict
from typing import Dict, Optional, TYPE_CHECKING, Tuple

import torch
import wx
from torch import Tensor

from app.components.utils import AnchorsChangedEvent, EVT_REFINING_ENDED, EVT_REFINING_STARTED
from crystalsizer3d.projector import ProjectedVertexKey

if TYPE_CHECKING:
    from app.components.image_panel import ImagePanel

ANCHOR_MODE_HIGHLIGHT = 'highlight'
ANCHOR_MODE_ANCHOR = 'anchor'
ANCHOR_MODE_VIEW = 'view'
ANCHOR_MODE_REMOVE = 'remove'
ANCHOR_MODES = [ANCHOR_MODE_HIGHLIGHT, ANCHOR_MODE_ANCHOR, ANCHOR_MODE_VIEW, ANCHOR_MODE_REMOVE]


class VertexNotFoundInImageError(Exception):
    pass


def _in_bounds(pos: Tensor):
    """
    Check if the position is within the image bounds.
    """
    return (-1 <= pos[0] <= 1) and (-1 <= pos[1] <= 1)


class AnchorManager:
    mode: str = ANCHOR_MODE_HIGHLIGHT
    highlighted_vertex: Optional[ProjectedVertexKey] = None
    selected_vertex: Optional[ProjectedVertexKey] = None
    anchor_point: Optional[Tensor] = None
    anchors: OrderedDict[ProjectedVertexKey, Tensor] = OrderedDict()
    anchor_visibility: Dict[ProjectedVertexKey, bool] = {}
    selected_anchor: Optional[ProjectedVertexKey] = None

    HIGHLIGHT_COLOUR_FACING = (255, 0, 0)
    HIGHLIGHT_COLOUR_BACK = (0, 0, 255)
    HIGHLIGHT_OUTLINE_ALPHA = 180
    HIGHLIGHT_FILL_ALPHA = 50
    HIGHLIGHT_CIRCLE_RADIUS = 12

    SELECTION_COLOUR_FACING = (255, 0, 0)
    SELECTION_COLOUR_BACK = (0, 0, 255)
    SELECTION_OUTLINE_ALPHA = 255
    SELECTION_FILL_ALPHA = 128
    SELECTION_CIRCLE_RADIUS = 9

    ANCHOR_LINE_COLOUR = (0, 0, 0)
    ANCHOR_CROSS_COLOUR = (0, 0, 0)
    ANCHOR_CROSS_SIZE = 10

    ACTIVE_COLOUR_FACING = (180, 0, 0)
    ACTIVE_COLOUR_BACK = (0, 0, 180)
    ACTIVE_OUTLINE_ALPHA = 180
    ACTIVE_FILL_ALPHA = 25
    ACTIVE_CIRCLE_RADIUS = 7
    ACTIVE_LINE_COLOUR = (40, 40, 40)
    ACTIVE_CROSS_COLOUR = (40, 40, 40)
    ACTIVE_CROSS_SIZE = 7

    def __init__(self, image_panel: 'ImagePanel'):
        self.image_panel = image_panel
        self._init_listeners()

    def __getattr__(self, item: str):
        """
        Try to get the attribute from the image_panel if it doesn't exist in the class
        """
        if hasattr(self.image_panel, item):
            return getattr(self.image_panel, item)

    def _init_listeners(self):
        """
        Initialise the event listeners.
        """
        self.image_containers['image'].Bind(wx.EVT_MOTION, self.on_mouse_hover)
        self.image_containers['image'].Bind(wx.EVT_LEFT_DOWN, self.on_mouse_click)
        self.image_windows['image'].Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
        self.app_frame.Bind(EVT_REFINING_STARTED, self.on_refining_started)
        self.app_frame.Bind(EVT_REFINING_ENDED, self.on_refining_ended)

    def _get_mouse_position(self) -> Tensor:
        """
        Get the relative position of the mouse within the image.
        """
        point = wx.GetMousePosition()
        point -= self.image_containers['image'].GetScreenPosition()
        x, y = point

        # Calculate the offsets between wireframe and container
        container_width, container_height = self.image_containers['image'].GetSize()
        img_width, img_height = self.bitmaps['wireframe'].GetSize()
        offset_x = (container_width - img_width) / 2
        offset_y = (container_height - img_height) / 2
        x -= offset_x
        y -= offset_y

        # Normalise coordinates relative to the centre of the image
        rel_x = (x - img_width / 2) / (img_width / 2)
        rel_y = (y - img_height / 2) / (img_height / 2)
        rel_y = -rel_y  # Flip y-axis to make it -1 at bottom and +1 at top
        pos = torch.tensor([rel_x, rel_y])

        return pos

    def _relative_to_image_coords(self, rel_coords: Tensor) -> Tuple[float, float]:
        """
        Convert relative coordinates to image coordinates.
        """
        width, height = self.bitmaps['wireframe'].GetSize()
        vx, vy = [
            float((rel_coords[0] + 1) * width / 2),
            float((1 - rel_coords[1]) * height / 2)
        ]
        return vx, vy

    def _get_vertex_image_coords(self, vertex_key: ProjectedVertexKey) -> Tuple[float, float]:
        """
        Get the projected image coordinates of a vertex.
        """
        if vertex_key not in self.projector.projected_vertex_keys:
            self.anchor_visibility[vertex_key] = False
            raise VertexNotFoundInImageError(f'Vertex {vertex_key} not found in the image.')
        self.anchor_visibility[vertex_key] = True
        idx = self.projector.projected_vertex_keys.index(vertex_key)
        v_coords_rel = self.projector.projected_vertex_coords[idx]
        vx, vy = self._relative_to_image_coords(v_coords_rel)
        return vx, vy

    def _new_canvas(self) -> wx.Bitmap:
        """
        Create a blank, transparent canvas to draw on.
        """
        width, height = self.bitmaps['wireframe'].GetSize()
        image = wx.Image(width, height)
        image.InitAlpha()
        alpha = [0] * (width * height)
        image.SetAlpha(bytes(alpha))
        canvas = wx.Bitmap(image)
        return canvas

    def _reset_to_highlight_mode(self):
        """
        Reset the anchor manager to initial highlight mode.
        """
        self.mode = ANCHOR_MODE_HIGHLIGHT
        self.highlighted_vertex = None
        self.selected_vertex = None
        self.selected_anchor = None
        self.anchor_point = None

    def on_mouse_hover(self, event: wx.MouseEvent):
        """
        Handle mouse hover events - either for highlighting of vertices or setting of points.
        """
        if self.crystal is None:
            return

        # Highlight the closest vertex to the mouse position
        if self.mode == ANCHOR_MODE_HIGHLIGHT:
            self.update_highlight(event)

        # Show a line from the selected vertex to the mouse position
        elif self.mode == ANCHOR_MODE_ANCHOR:
            self.update_anchor_point(event)

    def on_mouse_leave(self, event: wx.MouseEvent):
        """
        Remove any temporary overlays when the mouse leaves the area.
        """
        if self.mode == ANCHOR_MODE_HIGHLIGHT:
            self.highlighted_vertex = None

        elif self.mode == ANCHOR_MODE_ANCHOR:
            self.anchor_point = None

        self.update_overlay()

    def on_refining_started(self, event: wx.Event):
        """
        Disable the anchor manager while refining is in progress.
        """
        event.Skip()
        self._reset_to_highlight_mode()
        self.mode = ANCHOR_MODE_VIEW
        self.update_overlay()

    def on_refining_ended(self, event: wx.Event):
        """
        Re-enable the anchor manager when refining is complete.
        """
        event.Skip()
        self._reset_to_highlight_mode()
        self.update_overlay()

    def on_mouse_click(self, event: wx.MouseEvent):
        """
        Select the currently-highlighted vertex.
        """

        # Select the currently-highlighted vertex.
        if self.mode == ANCHOR_MODE_HIGHLIGHT:
            self.select_highlighted_vertex()

        # Fix the anchor point in the image or de-select the selected vertex.
        elif self.mode == ANCHOR_MODE_ANCHOR:
            pos = self._get_mouse_position()
            mx, my = self._relative_to_image_coords(pos)
            try:
                vx, vy = self._get_vertex_image_coords(self.selected_vertex)
            except VertexNotFoundInImageError:
                self._reset_to_highlight_mode()
                return

            # Close to the selected vertex so deselect it
            if math.sqrt((mx - vx) ** 2 + (my - vy) ** 2) < self.HIGHLIGHT_CIRCLE_RADIUS * 1.2:
                self.deselect_vertex()

            # Not too close to the selected vertex so fix the anchor point here
            else:
                self.fix_anchor_point(event)

    def update_highlight(self, event: wx.MouseEvent):
        """
        Highlight the vertex closest to the mouse position.
        """
        if self.mode != ANCHOR_MODE_HIGHLIGHT or self.bitmaps['wireframe'] is None:
            return
        pos = self._get_mouse_position()

        # Check if the coordinates are within the image bounds
        if not _in_bounds(pos):
            self.on_mouse_leave(event)
            return

        # Find the closest vertex to the mouse position and update the overlay
        vertex_dists = torch.cdist(pos[None, ...], self.projector.projected_vertex_coords)[0]
        closest_idx = vertex_dists.argmin()
        self.highlighted_vertex = self.projector.projected_vertex_keys[closest_idx]
        self.update_overlay()

    def select_highlighted_vertex(self):
        """
        Select the highlighted vertex.
        """
        if self.mode != ANCHOR_MODE_HIGHLIGHT or self.highlighted_vertex is None:
            return
        self.mode = ANCHOR_MODE_ANCHOR
        self.selected_vertex = self.highlighted_vertex
        self.highlighted_vertex = None
        self.update_overlay()

    def deselect_vertex(self):
        """
        Deselect the currently selected vertex.
        """
        if self.mode != ANCHOR_MODE_ANCHOR or self.selected_vertex is None:
            return
        self.mode = ANCHOR_MODE_HIGHLIGHT
        self.highlighted_vertex = self.selected_vertex
        self.selected_vertex = None
        self.update_overlay()

    def update_anchor_point(self, event: wx.MouseEvent):
        """
        Update the anchor target position.
        """
        if self.mode != ANCHOR_MODE_ANCHOR or self.selected_vertex is None:
            return
        pos = self._get_mouse_position()
        if not _in_bounds(pos):
            self.on_mouse_leave(event)
            return
        self.anchor_point = pos
        self.update_overlay()

    def fix_anchor_point(self, event: wx.MouseEvent):
        """
        Fix the anchor point in the image.
        """
        if self.mode != ANCHOR_MODE_ANCHOR or self.selected_vertex is None:
            return
        pos = self._get_mouse_position()
        if not _in_bounds(pos):
            return
        self.mode = ANCHOR_MODE_HIGHLIGHT
        self.anchors[self.selected_vertex] = pos
        self.anchor_visibility[self.selected_vertex] = True
        self.anchor_point = None
        self.selected_vertex = None
        self.selected_anchor = None
        self.update_overlay()
        wx.PostEvent(self.app_frame, AnchorsChangedEvent())

    def select_anchor(self, idx: Optional[int] = None):
        """
        Select one of the saved anchors.
        """
        if idx is None:
            self.mode = ANCHOR_MODE_HIGHLIGHT
            self.selected_anchor = None
        else:
            self.mode = ANCHOR_MODE_VIEW
            self.selected_anchor = list(self.anchors)[idx]
        self.update_overlay()

    def remove_anchor(self, idx: int):
        """
        Remove an anchor.
        """
        key = list(self.anchors)[idx]
        del self.anchors[key]
        del self.anchor_visibility[key]
        self._reset_to_highlight_mode()
        self.update_overlay()
        wx.PostEvent(self.app_frame, AnchorsChangedEvent())

    def remove_all_anchors(self):
        """
        Remove all anchors.
        """
        self.anchors.clear()
        self.anchor_visibility.clear()
        self._reset_to_highlight_mode()
        self.update_overlay()
        wx.PostEvent(self.app_frame, AnchorsChangedEvent())

    def update_overlay(self, update_images: bool = True):
        """
        Update the overlay image.
        """
        if self.bitmaps['wireframe'] is None:
            return
        canvas = self._new_canvas()

        # Create a graphics context from the bitmap
        mem_dc = wx.MemoryDC(canvas)
        gc = wx.GraphicsContext.Create(mem_dc)

        # Draw the active (saved) anchors
        for vertex_key, anchor_pos in self.anchors.items():
            if vertex_key not in self.anchor_visibility:
                self.anchor_visibility[vertex_key] = True
            was_visible = self.anchor_visibility[vertex_key]
            if self.selected_anchor == vertex_key:
                continue
            try:
                vx, vy = self._get_vertex_image_coords(vertex_key)
            except VertexNotFoundInImageError:
                if was_visible:
                    wx.PostEvent(self.app_frame, AnchorsChangedEvent())
                continue
            if not was_visible:
                wx.PostEvent(self.app_frame, AnchorsChangedEvent())
            ax, ay = self._relative_to_image_coords(anchor_pos)

            # Draw a line from the vertex to the anchor point
            gc.SetPen(wx.Pen((*self.ACTIVE_LINE_COLOUR, 150), 2, wx.PENSTYLE_SHORT_DASH))
            gc.StrokeLine(int(vx),int(vy),int(ax),int(ay))

            # Draw a circle at the location of the vertex
            colour = self.ACTIVE_COLOUR_FACING if vertex_key[1] == 'facing' else self.ACTIVE_COLOUR_BACK
            radius=self.ACTIVE_CIRCLE_RADIUS
            gc.SetPen(wx.Pen((*colour, self.ACTIVE_OUTLINE_ALPHA), 1))
            gc.SetBrush(wx.Brush((*colour, self.ACTIVE_FILL_ALPHA)))
            gc.DrawEllipse(vx - radius, vy - radius, 2 * radius, 2 * radius)

            # Draw a cross at the location of the anchor point
            d = self.ANCHOR_CROSS_SIZE / math.sqrt(2)
            gc.SetPen(wx.Pen((*self.ACTIVE_CROSS_COLOUR, 150), 1))
            gc.StrokeLine(int(ax - d), int(ay - d), int(ax + d), int(ay + d))
            gc.StrokeLine(int(ax + d), int(ay - d), int(ax - d), int(ay + d))


        # Draw a cross at the selected anchor position with a connecting line from the selected vertex
        if self.selected_anchor is not None or self.selected_vertex is not None and self.anchor_point is not None:
            if self.selected_anchor is not None:
                vertex_key = self.selected_anchor
                anchor_point = self.anchors[self.selected_anchor]
            else:
                vertex_key = self.selected_vertex
                anchor_point = self.anchor_point
            try:
                vx, vy = self._get_vertex_image_coords(vertex_key)
                ax, ay = self._relative_to_image_coords(anchor_point)
                draw_line = math.sqrt((ax - vx) ** 2 + (ay - vy) ** 2) > self.HIGHLIGHT_CIRCLE_RADIUS * 1.2

                if draw_line:
                    # Connecting line
                    gc.SetPen(wx.Pen((*self.ANCHOR_LINE_COLOUR, 255), 2, wx.PENSTYLE_SHORT_DASH))
                    gc.StrokeLine(int(vx),int(vy),int(ax),int(ay))

                    # Cross at the anchor point
                    d = self.ANCHOR_CROSS_SIZE / math.sqrt(2)
                    gc.SetPen(wx.Pen((*self.ANCHOR_CROSS_COLOUR, 255), 2))
                    gc.StrokeLine(int(ax - d), int(ay - d), int(ax + d), int(ay + d))
                    gc.StrokeLine(int(ax + d), int(ay - d), int(ax - d), int(ay + d))


            # If the vertex is not found, reset to highlight mode unless we were trying to show a saved anchor
            except VertexNotFoundInImageError:
                if self.selected_anchor is None:
                    return self._reset_to_highlight_mode()

        # Draw a circle at the location of the highlighted/selected vertex
        if ((self.mode == ANCHOR_MODE_HIGHLIGHT and self.highlighted_vertex is not None)
                or self.selected_vertex is not None or self.selected_anchor is not None):
            if self.mode == ANCHOR_MODE_HIGHLIGHT:
                v_id, face_idx = self.highlighted_vertex
                colour = self.HIGHLIGHT_COLOUR_FACING if face_idx == 'facing' else self.HIGHLIGHT_COLOUR_BACK
                outline_colour = (*colour, self.HIGHLIGHT_OUTLINE_ALPHA)
                fill_colour = (*colour, self.HIGHLIGHT_FILL_ALPHA)
                radius = self.HIGHLIGHT_CIRCLE_RADIUS
            else:
                if self.selected_anchor is not None:
                    v_id, face_idx = self.selected_anchor
                else:
                    v_id, face_idx = self.selected_vertex
                colour = self.SELECTION_COLOUR_FACING if face_idx == 'facing' else self.SELECTION_COLOUR_BACK
                outline_colour = (*colour, self.SELECTION_OUTLINE_ALPHA)
                fill_colour = (*colour, self.SELECTION_FILL_ALPHA)
                radius = self.SELECTION_CIRCLE_RADIUS

            try:
                vx, vy = self._get_vertex_image_coords((v_id, face_idx))

                gc.SetPen(wx.Pen(outline_colour, 1))
                gc.SetBrush(wx.Brush(fill_colour))
                gc.DrawEllipse(vx - radius, vy - radius, 2 * radius, 2 * radius)

            # If the vertex is not found, reset to highlight mode unless we were trying to show a saved anchor
            except VertexNotFoundInImageError:
                if self.selected_anchor is None:
                    return self._reset_to_highlight_mode()

        mem_dc.SelectObject(wx.NullBitmap)
        self.images['anchors'] = canvas.ConvertToImage()
        if update_images:
            self.update_images(quiet=True)
