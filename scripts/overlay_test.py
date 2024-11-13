import matplotlib.pyplot as plt
import wx

import cv2

from app.components.utils import wx_image_to_numpy

HIGHLIGHT_COLOUR_FACING = (255, 0, 0)
HIGHLIGHT_COLOUR_BACK = (0, 0, 255)
HIGHLIGHT_OUTLINE_ALPHA = 180
HIGHLIGHT_FILL_ALPHA = 50
HIGHLIGHT_CIRCLE_RADIUS = 12

HIGHLIGHT_COLOUR_OUTLINE = (255, 0, 0, 255)
HIGHLIGHT_COLOUR_FILL = (255, 0, 0, 255)


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
    # outline_colour = (*colour, HIGHLIGHT_OUTLINE_ALPHA)
    # fill_colour = (*colour, HIGHLIGHT_FILL_ALPHA)
    """
    The MemoryDC methods won't recognize the alpha channel.
    Even if a solid colour is given, the canvas is still painted with transparent colour.
    Consider use other ways of painting.
    Check wxPython docs - wx.DC
    """
    outline_colour = HIGHLIGHT_COLOUR_OUTLINE
    fill_colour = HIGHLIGHT_COLOUR_FILL
    radius = HIGHLIGHT_CIRCLE_RADIUS
    vx, vy = 0, 0

    mem_dc.SetPen(wx.Pen(outline_colour, 1))
    mem_dc.SetBrush(wx.Brush(fill_colour))
    mem_dc.DrawCircle(vx, vy, radius)

    mem_dc.SelectObject(wx.NullBitmap)
    image = canvas.ConvertToImage()
    npimg = wx_image_to_numpy(image)
    print(npimg)
    plt.imshow(npimg)
    plt.show()


def make_overlay_1():
    """
    Make an overlay image.
    Use cv2 to paint on numpy image?
    """
    canvas = _new_canvas()
    image = canvas.ConvertToImage()
    npimg = wx_image_to_numpy(image)

    colour = HIGHLIGHT_COLOUR_FACING
    outline_colour = (*colour, HIGHLIGHT_OUTLINE_ALPHA)
    fill_colour = (*colour, HIGHLIGHT_FILL_ALPHA)

    radius = HIGHLIGHT_CIRCLE_RADIUS
    vx, vy = 0, 0

    npimg = cv2.circle(npimg, [vx,vy], radius, outline_colour, 2)
    npimg = cv2.circle(npimg, [vx, vy], radius, fill_colour, -1)
    (b,g,r,a)=cv2.split(npimg)

    image=wx.Image(len(npimg),len(npimg[0]),cv2.merge((b,g,r)),a)
    npimg2=wx_image_to_numpy(image)

    print(npimg2)
    plt.imshow(npimg2)
    plt.show()

def make_overlay_2():
    """
    Make an overlay image.
    Use wx.GraphicsContext instead of wx.DC
    """
    canvas = _new_canvas()

    # Create a graphics context from the bitmap
    mem_dc = wx.MemoryDC(canvas)
    gc = wx.GraphicsContext.Create(mem_dc)

    # Set the pen and brush with the specified colours and alpha
    colour = HIGHLIGHT_COLOUR_FACING
    outline_colour = (*colour, HIGHLIGHT_OUTLINE_ALPHA)
    fill_colour = (*colour, HIGHLIGHT_FILL_ALPHA)
    pen = gc.CreatePen(wx.Pen(outline_colour, 1))
    brush = gc.CreateBrush(wx.Brush(fill_colour))

    gc.SetPen(pen)
    gc.SetBrush(brush)

    radius = HIGHLIGHT_CIRCLE_RADIUS
    vx, vy = 0, 0

    # Draw the circle with the graphics context
    gc.DrawEllipse(vx - radius, vy - radius, 2 * radius, 2 * radius)

    # Clean up
    mem_dc.SelectObject(wx.NullBitmap)
    image = canvas.ConvertToImage()
    npimg = wx_image_to_numpy(image)
    print(npimg)
    plt.imshow(npimg)
    plt.show()


class TestApp(wx.App):
    pass


if __name__ == '__main__':
    app = TestApp()
    make_overlay_2()
