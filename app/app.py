import wx

from app import APP_SETTINGS_PATH
from app.components.app_frame import AppFrame



class CrystalSizerApp(wx.App):
    config: wx.Config = None
    frame: AppFrame = None

    def OnInit(self):
        # Load the configuration
        self.config = wx.Config('CrystalSizer3D', 'University of Leeds',
                                localFilename=str(APP_SETTINGS_PATH / 'config.ini'))

        # Load the frame and start the app
        self.frame = AppFrame(self.config)
        self.frame.Show()
        # self.frame.load_state()
        # wx.CallAfter(self.frame.load_state)
        wx.FutureCall(0, self.frame.load_state)  # Wait until the GUI has been drawn fully
        return True

    def OnExit(self):
        self.config.Flush()  # Ensure settings are saved before exiting
        return 0


if __name__ == '__main__':
    app = CrystalSizerApp()  # False)
    app.MainLoop()
