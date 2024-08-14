import matplotlib.pyplot as plt
import wx

from app.components.utils import wx_image_to_numpy

HIGHLIGHT_COLOUR_FACING = (255, 0, 0)
HIGHLIGHT_COLOUR_BACK = (0, 0, 255)
HIGHLIGHT_OUTLINE_ALPHA = 180
HIGHLIGHT_FILL_ALPHA = 50
HIGHLIGHT_CIRCLE_RADIUS = 12


def _new_canvas() -> wx.Bitmap:
    """
    Create a blank, transparent canvas to draw on.
    """
    width, height = 300, 300
    image = wx.Image(width, height)
    image.InitAlpha()
    alpha = [0] * (width * height)
    image.SetAlpha(bytes(alpha))
    canvas = wx.Bitmap(image)
    return canvas


def make_overlay():
    """
    Make an overlay image.
    """
    canvas = _new_canvas()
    mem_dc = wx.MemoryDC()
    mem_dc.SelectObject(canvas)

    colour = HIGHLIGHT_COLOUR_FACING
    outline_colour = (*colour, HIGHLIGHT_OUTLINE_ALPHA)
    fill_colour = (*colour, HIGHLIGHT_FILL_ALPHA)
    radius = HIGHLIGHT_CIRCLE_RADIUS
    vx, vy = 150, 150
    mem_dc.SetPen(wx.Pen(outline_colour, 1))
    mem_dc.SetBrush(wx.Brush(fill_colour))
    mem_dc.DrawCircle(vx, vy, radius)

    mem_dc.SelectObject(wx.NullBitmap)
    image = canvas.ConvertToImage()
    npimg = wx_image_to_numpy(image)
    plt.imshow(npimg)
    plt.show()


class DummyApp(wx.App):
    pass


if __name__ == '__main__':
    app = DummyApp()
    make_overlay()
