from abc import ABCMeta, abstractmethod
from pathlib import Path

import wx

from crystalsizer3d.crystal import Crystal


class AppPanelMeta(ABCMeta, type(wx.Panel)):
    pass


class AppPanel(wx.Panel, metaclass=AppPanelMeta):
    """
    Abstract base class for all panels in the application.
    """

    def __init__(self, app_frame: 'AppFrame' = None):
        self.app_frame = app_frame
        super().__init__(parent=app_frame)

        # Initialise components
        self._init_components()

        # Add event listeners
        self._init_listeners()

    @property
    def config(self):
        return self.GetParent().config

    @property
    def crystal(self) -> Crystal:
        return self.app_frame.crystal

    @property
    def image_path(self) -> Path:
        return self.app_frame.image_path

    def _log(self, message: str):
        """
        Log a message to the status bar and to the logger.
        """
        self.app_frame._log(message)

    @abstractmethod
    def _init_components(self):
        """
        Initialise the components of the panel.
        """
        pass

    def _init_listeners(self):
        """
        Initialise the event listeners for the panel.
        """
        pass
