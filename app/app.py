import atexit
import signal
import sys

import wx

from app import APP_DATA_PATH
from app.components.app_frame import AppFrame
from app.components.parallelism import close_parallel_processes


class CrystalSizerApp(wx.App):
    config: wx.Config = None
    frame: AppFrame = None

    def OnInit(self):
        APP_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Load the configuration
        self.config = wx.Config('CrystalSizer3D', 'University of Leeds',
                                localFilename=str(APP_DATA_PATH / 'config.ini'))

        # Load the frame and start the app
        self.frame = AppFrame(self.config)
        self.frame.Show()
        self.frame.Maximize(True)
        wx.CallLater(0, self.frame.load_state)  # Wait until the GUI has been drawn fully
        return True

    def OnExit(self):
        self.config.Flush()  # Ensure settings are saved before exiting
        close_parallel_processes()
        return 0


def signal_handler(sig, frame):
    """Handle termination signals."""
    close_parallel_processes()
    sys.exit(0)


# Register cleanup function to be called on exit
atexit.register(close_parallel_processes)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    app = CrystalSizerApp()  # False)
    app.MainLoop()
