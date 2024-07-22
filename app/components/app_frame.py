from pathlib import Path

import torch
import wx

from app import CRYSTAL_DATA_PATH
from app.components.control_panel import ControlPanel
from app.components.image_panel import ImagePanel
from app.components.utils import CrystalChangedEvent, EVT_CRYSTAL_CHANGED, ImagePathChangedEvent
from crystalsizer3d import logger
from crystalsizer3d.crystal import Crystal


class AppFrame(wx.Frame):
    crystal: Crystal = None
    image_path: Path = None

    def __init__(self, config: wx.Config):
        super().__init__(parent=None, title='Crystal Sizer 3D', size=(1280, 720))
        self.config = config
        self.SetWindowStyle(style=wx.DEFAULT_DIALOG_STYLE | wx.MINIMIZE_BOX)

        # Initialise panels
        self.control_panel = ControlPanel(self)
        self.image_panel = ImagePanel(self)
        self._init_optimisation_panel()

        # Put panels together
        base_sizer = wx.BoxSizer(wx.HORIZONTAL)
        base_sizer.Add(self.control_panel, 1, wx.EXPAND)
        base_sizer.Add(self.image_panel, 3, wx.EXPAND)
        base_sizer.Add(self.control_panel_R, 1, wx.EXPAND)
        self.SetSizer(base_sizer)

        # Initialise status bar
        self.CreateStatusBar()
        self.SetStatusText('CrystalSizer3D')

        # Add event listeners
        self.Bind(EVT_CRYSTAL_CHANGED, self.update_crystal_json)

    def _log(self, message: str):
        """
        Log a message to the status bar and to the logger.
        """
        logger.info(message)
        self.SetStatusText(message)

    def load_state(self):
        """
        Load the application state.
        """
        image_path = self.config.Read('image_path')
        if image_path != '':
            image_path = Path(image_path)
            if image_path.exists():
                self.image_path = image_path
                wx.CallAfter(lambda: wx.PostEvent(self, ImagePathChangedEvent()))

        if CRYSTAL_DATA_PATH.exists():
            self.crystal = Crystal.from_json(CRYSTAL_DATA_PATH)
            wx.CallAfter(lambda: wx.PostEvent(self, CrystalChangedEvent()))

    def update_crystal_json(self, event: wx.Event):
        """
        Update the crystal json file.
        """
        self.crystal.to_json(CRYSTAL_DATA_PATH, overwrite=True)
        event.Skip()

    def _init_optimisation_panel(self):
        """
        Initialize optimisation panel.
        """
        self.control_panel_R = wx.Panel(self)
        ctrlPnlRSizer = wx.BoxSizer(wx.VERTICAL)

        self.optimTxt = wx.StaticText(self.control_panel_R, label='Mesh Optimizer')
        ctrlPnlRSizer.Add(self.optimTxt, 0, wx.EXPAND | wx.ALL, 5)

        # Optimizer restrictions
        self.restrcList = wx.ListCtrl(self.control_panel_R, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.restrcList.SetMinSize(wx.Size(256, 256))
        self.restrcList.InsertColumn(0, 'Vertex')
        self.restrcList.SetColumnWidth(col=0, width=100)
        self.restrcList.InsertColumn(1, 'Coordinate 2D')
        self.restrcList.SetColumnWidth(col=1, width=100)
        ctrlPnlRSizer.Add(self.restrcList, 0, wx.EXPAND | wx.ALL, 5)

        # Optimizer button (temporary placeholder)
        self.btnOptim = wx.Button(self.control_panel_R, label='Do not press')
        self.btnOptim.Bind(wx.EVT_BUTTON, self.on_optim)
        ctrlPnlRSizer.Add(self.btnOptim, 0, wx.EXPAND | wx.ALL, 5)
        self.control_panel_R.SetSizer(ctrlPnlRSizer)

    def on_optim(self, event):
        print('> vertices')
        print(self.crystal.vertices)
        print(self.vertices_2d)
        print('> faces')
        print(self.crystal.faces)

        # # One time use only - generate target positions from fitted crystal
        # # 24/06/24 - ImageBitmap size change - Remember to redo all coordinates!
        # dctlst = []
        # for i in range(len(self.crystal.vertices)):
        #     fcslst = []
        #     for mlr_idx in self.crystal.faces:
        #         if i in self.crystal.faces[mlr_idx].tolist():
        #             fcslst.append(mlr_idx)
        #     dctlst.append({'faces': fcslst, 'position_2d': self.vertices_2d[i].tolist()})
        # print(dctlst)

        # Fitted crystal - alpha4.json
        target_positions = [
            {'faces': [(0, 0, 1), (1, 1, 1), (1, -1, 1)], 'position_2d': [460.37896728515625, 409.05523681640625]},
            {'faces': [(0, 0, 1), (1, 1, 1), (0, 1, 1)], 'position_2d': [344.228759765625, 360.9009704589844]},
            {'faces': [(0, 0, 1), (1, -1, 1), (0, -1, 1)], 'position_2d': [542.4766845703125, 322.4871520996094]},
            {'faces': [(0, 0, 1), (-1, 1, 1), (-1, -1, 1)], 'position_2d': [412.1661376953125, 260.7255859375]},
            {'faces': [(0, 0, 1), (-1, 1, 1), (0, 1, 1)], 'position_2d': [339.2733154296875, 337.5874938964844]},
            {'faces': [(0, 0, 1), (-1, -1, 1), (0, -1, 1)], 'position_2d': [540.6732788085938, 314.00286865234375]},
            {'faces': [(0, 0, -1), (1, 1, -1), (1, -1, -1)], 'position_2d': [456.6241760253906, 419.4239807128906]},
            {'faces': [(0, 0, -1), (1, 1, -1), (0, 1, -1)], 'position_2d': [351.35150146484375, 375.77935791015625]},
            {'faces': [(0, 0, -1), (1, -1, -1), (0, -1, -1)], 'position_2d': [546.0210571289062, 325.1593017578125]},
            {'faces': [(0, 0, -1), (-1, 1, -1), (-1, -1, -1)], 'position_2d': [425.3760681152344, 249.98631286621094]},
            {'faces': [(0, 0, -1), (-1, 1, -1), (0, 1, -1)], 'position_2d': [343.0623779296875, 336.7821044921875]},
            {'faces': [(0, 0, -1), (-1, -1, -1), (0, -1, -1)], 'position_2d': [540.1574096679688, 297.5730895996094]},
            {'faces': [(1, 1, 1), (1, 1, -1), (1, 0, 0)], 'position_2d': [452.66888427734375, 429.6434631347656]},
            {'faces': [(1, 1, 1), (1, 1, -1), (0, 1, -1)], 'position_2d': [332.44146728515625, 379.7987976074219]},
            {'faces': [(1, 1, 1), (1, -1, 1), (1, 0, 0)], 'position_2d': [464.232666015625, 427.18548583984375]},
            {'faces': [(1, 1, 1), (0, 1, 1), (0, 1, -1)], 'position_2d': [317.7055358886719, 366.5386657714844]},
            {'faces': [(1, 1, -1), (1, -1, -1), (1, 0, 0)], 'position_2d': [458.53155517578125, 428.3973083496094]},
            {'faces': [(1, -1, 1), (1, -1, -1), (1, 0, 0)], 'position_2d': [470.0953063964844, 425.9393615722656]},
            {'faces': [(1, -1, 1), (1, -1, -1), (0, -1, -1)], 'position_2d': [570.6322021484375, 319.92803955078125]},
            {'faces': [(1, -1, 1), (0, -1, 1), (0, -1, -1)], 'position_2d': [572.3182983398438, 316.1440734863281]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (-1, 0, 0)], 'position_2d': [405.86444091796875, 245.04063415527344]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (0, 1, 1), (0, 1, -1)],
             'position_2d': [312.7500915527344, 343.2251892089844]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (0, 1, 1), (0, 1, -1)],
             'position_2d': [312.7500915527344, 343.2251892089844]},
            {'faces': [(-1, 1, 1), (-1, -1, 1), (-1, 0, 0)], 'position_2d': [408.7038879394531, 244.43707275390625]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (0, 1, 1), (0, 1, -1)],
             'position_2d': [312.7500915527344, 343.2251892089844]},
            {'faces': [(-1, 1, -1), (-1, -1, -1), (-1, 0, 0)], 'position_2d': [423.5268249511719, 241.286376953125]},
            {'faces': [(-1, 1, 1), (-1, 1, -1), (0, 1, 1), (0, 1, -1)],
             'position_2d': [312.7500915527344, 343.2251892089844]},
            {'faces': [(-1, -1, 1), (-1, -1, -1), (-1, 0, 0)], 'position_2d': [426.3663024902344, 240.6828155517578]},
            {'faces': [(-1, -1, 1), (-1, -1, -1), (0, -1, -1)], 'position_2d': [555.646728515625, 294.2807312011719]},
            {'faces': [(-1, -1, 1), (0, -1, 1), (0, -1, -1)], 'position_2d': [570.514892578125, 307.6598815917969]}]

        # Optimizer from Tom I
        opt = torch.optim.Adam([
            self.crystal.distances
        ], lr=0.01)

        for i in range(100):
            opt.zero_grad()
            self.update_crystal(None)
            # self.crystal.update_mesh()
            loss = 0
            for target_position in target_positions:

                # Find the joining vertex from faces - index is 'x'
                # Continue if not found or find more than 1
                xf = list(range(0, len(self.crystal.vertices)))
                for face in target_position['faces']:
                    if face in self.crystal.faces:
                        xf = list(set(xf).intersection(set(self.crystal.faces[face].tolist())))
                    else:
                        continue
                if len(xf) != 1:
                    continue
                x = xf[0]

                loss = loss + torch.norm(
                    self.vertices_2d[x] - torch.tensor(target_position['position_2d'])
                )
            loss.backward()
            print(f'Step {i}, Loss: {loss.item()}, Distances: {self.crystal.distances}')

            # Update distances in list
            for t in range(0, len(self.crystal.distances)):
                self.face_list.SetItem(t, 1, str(format(self.crystal.distances[t], '.2f')))

            # Save image of every step
            filename = 'Step_' + '{:03d}'.format(i) + '.jpg'
            bitmap = self.img_bitmap.GetBitmap()
            bitmap.SaveFile('../TestData_Optimizer_2/demo/optimSteps/' + filename, wx.BITMAP_TYPE_JPEG)

            opt.step()
