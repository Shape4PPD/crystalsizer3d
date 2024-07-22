import numpy as np
import wx
import wx.lib.newevent

CrystalChangedEvent, EVT_CRYSTAL_CHANGED = wx.lib.newevent.NewEvent()
ImagePathChangedEvent, EVT_IMAGE_PATH_CHANGED = wx.lib.newevent.NewEvent()


def wx_image_to_numpy(image: wx.Image) -> np.ndarray:
    """
    Convert a wx.Image to a numpy array.
    """
    width, height = image.GetWidth(), image.GetHeight()
    image_data = image.GetData()
    np_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 3))
    return np_array


def numpy_to_wx_image(image_array: np.ndarray) -> wx.Image:
    """
    Convert a numpy array to a wx.Image.
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError('Numpy array must be of shape (height, width, 3)')
    h, w = image_array.shape[:2]
    image_array = image_array.astype(np.uint8)
    image_data = image_array.tobytes()
    wx_image = wx.Image(w, h)
    wx_image.SetData(image_data)
    return wx_image
