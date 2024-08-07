from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import wx

from app.components.refiner_proxy import RefinerProxy
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.projector import Projector
from crystalsizer3d.scene_components.scene import Scene

if TYPE_CHECKING:
    from app.components.app_frame import AppFrame


class AppPanelMeta(ABCMeta, type(wx.Panel)):
    pass


class AppPanel(wx.Panel, metaclass=AppPanelMeta):
    """
    Abstract base class for all panels in the application.
    """

    def __init__(self, app_frame: 'AppFrame' = None):
        self.app_frame: AppFrame = app_frame
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
    def refiner(self) -> RefinerProxy:
        if self.app_frame.refiner is None:
            self.app_frame.init_refiner()
        return self.app_frame.refiner

    @property
    def scene(self) -> Scene:
        if self.app_frame.scene is None:
            self.app_frame.init_scene()
        return self.app_frame.scene

    @property
    def projector(self) -> Projector:
        if self.app_frame.projector is None:
            self.app_frame.init_projector()
        return self.app_frame.projector

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
