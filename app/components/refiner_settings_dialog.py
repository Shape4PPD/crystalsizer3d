from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from decimal import Decimal, getcontext
from pathlib import Path
from typing import List, Optional, Tuple, Union

import wx
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
from wx.lib.scrolledpanel import ScrolledPanel

from app import REFINER_ARGS_PATH
from crystalsizer3d.args.refiner_args import RefinerArgs

CATEGORY_SELECTOR_HEADING_FONT_COLOUR = '#be3a3a'
CATEGORY_SELECTOR_FONT_COLOUR = '#ffffff'
CATEGORY_SELECTOR_BACKGROUND_COLOUR = '#555555'
CATEGORY_HEADING_FONT_COLOUR = '#0068c6'

getcontext().prec = 50  # Set high precision for Decimal to avoid rounding errors


def var_to_arg(var_name) -> str:
    return f"--{var_name.replace('_', '-')}"


def arg_to_var(arg_name) -> str:
    return arg_name.lstrip('-').replace('-', '_')


def is_floatable(value) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _get_font(bold: bool = True):
    """
    Get the default GUI font.
    """
    font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
    if bold:
        font.SetWeight(wx.BOLD)
    return font


class ScientificNumberCtrl(wx.TextCtrl):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.Bind(wx.EVT_KILL_FOCUS, self.format_scientific)
        self.format_scientific()

    def format_scientific(self, event=None):
        value = self.GetValue()
        if not is_floatable(value):
            value = '0.0'
        float_value = Decimal(value)
        base, exponent = f'{float_value:.20e}'.split('e')
        base = base.rstrip('0').rstrip('.')
        if exponent in ['+0', '-0', '0'] or base == '0':
            self.SetValue(f'{base}')
        else:
            if base == '':
                base = '0'
            self.SetValue(f'{base}e{exponent}')
        self.Refresh()
        if event is not None:
            event.Skip()


class SettingsPanelMeta(ABCMeta, type(wx.Panel)):
    pass


class SettingsPanel(wx.Panel, metaclass=SettingsPanelMeta):
    """
    Base class for settings panels. Sets up a scrollable panel with a static box heading.
    """
    HEADING = ''

    def __init__(self, parent: wx.Panel, refiner_args: RefinerArgs):
        wx.Panel.__init__(self, parent)

        # Get the command line options out of the refiner args
        self.refiner_args = refiner_args
        self.refiner_args_parser = ArgumentParser()
        RefinerArgs.add_args(self.refiner_args_parser)

        # Create the static box container and heading
        container = wx.StaticBox(self, label=self.HEADING, style=wx.BORDER_NONE)
        container.SetForegroundColour(CATEGORY_HEADING_FONT_COLOUR)
        container.SetFont(_get_font())

        # Make the scrollable field container and sizer
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scrolled_panel = ScrolledPanel(self)
        self.scrolled_panel.SetSizer(self.scroll_sizer)

        # Add the fields to the panel
        self._init_fields()

        # Set up scrolling
        self.scrolled_panel.SetupScrolling(scroll_x=False, scroll_y=True, rate_y=10)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        scroll_sizer = wx.StaticBoxSizer(container, wx.VERTICAL)
        scroll_sizer.Add(self.scrolled_panel, 1, wx.EXPAND | wx.TOP, 10)
        sizer.Add(scroll_sizer, 1, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(sizer)
        sizer.Layout()

    @abstractmethod
    def _init_fields(self):
        """
        Initialise the fields for the panel.
        """
        pass

    def add_field(
            self,
            key: str,
            label: str,
            field_type: str,
            choices: Optional[List[str]] = None,
            min_value: Optional[Union[float, int]] = None,
            max_value: Optional[Union[float, int]] = None,
            increment: Optional[Union[float, int]] = None,
            file_picker_wildcard: Optional[str] = None,
            file_picker_message: Optional[str] = None

    ) -> Tuple[wx.StaticText, wx.Control, wx.StaticText]:
        """
        Add a field to the panel.
        """
        assert hasattr(self.refiner_args, key), f'Unknown field: {key}'
        value = getattr(self.refiner_args, key)

        # Label
        label = wx.StaticText(self.scrolled_panel, label=label)

        # Field
        if field_type == 'choice':
            assert choices is not None and len(choices) > 0, 'Choices must be provided for a choice field.'
            field = wx.Choice(self.scrolled_panel, choices=choices)
            field.SetSelection(choices.index(str(value)))

        elif field_type == 'bool':
            field = wx.CheckBox(self.scrolled_panel)
            field.SetValue(bool(value))

        elif field_type == 'text':
            value = str(value) if value is not None else ''
            field = wx.TextCtrl(self.scrolled_panel, value=value)

        elif field_type == 'int':
            value = str(value) if value is not None else ''
            ctrl_args = {'value': value}
            if min_value is not None:
                ctrl_args['min'] = min_value
            if max_value is not None:
                ctrl_args['max'] = max_value
            field = wx.SpinCtrl(self.scrolled_panel, **ctrl_args)

        elif field_type == 'float':
            value = str(value) if value is not None else ''
            ctrl_args = {'value': value}
            if min_value is not None:
                ctrl_args['min'] = min_value
            if max_value is not None:
                ctrl_args['max'] = max_value
            if increment is not None:
                ctrl_args['inc'] = increment
            field = wx.SpinCtrlDouble(self.scrolled_panel, **ctrl_args)

        elif field_type == 'scientific':
            value = str(value) if value is not None else ''
            field = ScientificNumberCtrl(self.scrolled_panel, value=value)

        elif field_type == 'path':
            value = str(value) if value is not None else ''
            ctrl_args = {'path': value}
            if file_picker_wildcard is not None:
                ctrl_args['wildcard'] = file_picker_wildcard
            if file_picker_message is not None:
                ctrl_args['message'] = file_picker_message
            field = wx.FilePickerCtrl(self.scrolled_panel, **ctrl_args)
            path_value = 'No file selected' if value == '' else value
            path_display = wx.StaticText(self.scrolled_panel, label=path_value)
            font = path_display.GetFont()
            font.SetWeight(wx.BOLD)
            path_display.SetFont(font)

            def on_path_change(event):
                path_display.SetLabelText(field.GetPath())
                event.Skip()

            field.Bind(wx.EVT_FILEPICKER_CHANGED, on_path_change)

        else:
            raise ValueError(f'Unknown field type: {field_type}')

        # Help text - pull from the argparse help
        help_text_str = self.refiner_args_parser._option_string_actions[var_to_arg(key)].help
        help_text = wx.StaticText(self.scrolled_panel, label=help_text_str)
        font = help_text.GetFont()
        font.SetStyle(wx.FONTSTYLE_ITALIC)
        help_text.SetFont(font)

        # Create a horizontal sizer for field and help
        inline_sizer = wx.BoxSizer(wx.HORIZONTAL)
        inline_sizer.Add(field, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)
        inline_sizer.Add(help_text, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)

        # Add to the sizer
        self.scroll_sizer.Add(label, flag=wx.EXPAND | wx.ALL)
        if field_type == 'path':
            self.scroll_sizer.Add(path_display, flag=wx.EXPAND | wx.ALL, border=5)
        self.scroll_sizer.Add(inline_sizer, flag=wx.EXPAND | wx.ALL, border=5)
        self.scroll_sizer.AddSpacer(20)

        def on_resize(event=None):
            """
            Resize the help text when the field is resized.
            """
            width = self.scrolled_panel.GetSize().GetWidth()
            field_width = field.GetSize().GetWidth()
            help_width = max(50, width - field_width - 40)
            help_text.SetMinSize((help_width, -1))
            help_text.SetLabelText(help_text_str)
            help_text.Wrap(help_width)
            self.Layout()
            if event is not None:
                event.Skip()

        self.Bind(wx.EVT_SIZE, on_resize)

        # Toggle checkboxes with a click on the label or help text
        if field_type == 'bool':

            def click_label(event):
                if not field.IsEnabled():
                    return
                field.SetValue(not field.GetValue())
                checkbox_event = wx.CommandEvent(wx.EVT_CHECKBOX.typeId, field.GetId())
                field.GetEventHandler().ProcessEvent(checkbox_event)
                event.Skip()

            label.Bind(wx.EVT_LEFT_DOWN, click_label)
            help_text.Bind(wx.EVT_LEFT_DOWN, click_label)

        def update_refiner_args(event):
            # Extract the new value from the field
            if field_type == 'choice':
                new_value = choices[field.GetSelection()]
            elif field_type in ['bool', 'text']:
                new_value = field.GetValue()
            elif field_type == 'int':
                new_value = int(field.GetValue())
            elif field_type in ['float', 'scientific']:
                new_value = float(field.GetValue())
                # new_value = Decimal(new_value)
            elif field_type == 'path':
                new_value = field.GetPath()
            else:
                raise ValueError(f'Unknown field type: {field_type}')

            # Fix the type of the new value
            old_value = getattr(self.refiner_args, key)
            old_type = type(old_value)
            new_type = type(new_value)
            if old_type is not None and old_type is not new_type:
                if old_type is ScalarFloat:
                    old_type = float
                new_value = old_type(new_value)

            # Update the refiner args
            setattr(self.refiner_args, key, new_value)
            event.Skip()

        # Update the refiner args when the field changes
        field.Bind(wx.EVT_KILL_FOCUS, update_refiner_args)
        field.Bind(wx.EVT_CHOICE, update_refiner_args)
        field.Bind(wx.EVT_CHECKBOX, update_refiner_args)
        field.Bind(wx.EVT_SPINCTRL, update_refiner_args)
        field.Bind(wx.EVT_SPINCTRLDOUBLE, update_refiner_args)

        return label, field, help_text


class InitialPredictionSettings(SettingsPanel):
    HEADING = 'Initial prediction'

    def _init_fields(self):
        self.add_field(
            'predictor_model_path',
            'Predictor model path',
            'path',
            file_picker_wildcard='JSON files (*.json)|*.json',
        )
        self.add_field(
            'initial_pred_from',
            'Initial prediction from',
            'choice',
            choices=['denoised', 'original']
        )
        self.add_field(
            'initial_pred_batch_size',
            'Initial prediction batch size',
            'choice',
            choices=['4', '8', '16', '32']
        )
        _, initial_pred_noise_min, _ = self.add_field(
            'initial_pred_noise_min',
            'Initial prediction noise min',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )
        _, initial_pred_noise_max, _ = self.add_field(
            'initial_pred_noise_max',
            'Initial prediction noise max',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )

        # Validate the noise min and max
        def validate_noise_min_max(event):
            min_value = initial_pred_noise_min.GetValue()
            max_value = initial_pred_noise_max.GetValue()
            if min_value > max_value:
                initial_pred_noise_min.SetValue(max_value)
                wx.PostEvent(initial_pred_noise_min,
                             wx.CommandEvent(wx.EVT_SPINCTRLDOUBLE.typeId, initial_pred_noise_min.GetId()))
            event.Skip()

        initial_pred_noise_min.Bind(wx.EVT_SPINCTRLDOUBLE, validate_noise_min_max)
        initial_pred_noise_max.Bind(wx.EVT_SPINCTRLDOUBLE, validate_noise_min_max)


class DenoiserSettings(SettingsPanel):
    HEADING = 'Denoiser'

    def _init_fields(self):
        self.add_field(
            'denoiser_model_path',
            'Denoiser model path',
            'path',
            file_picker_wildcard='JSON files (*.json)|*.json',
        )
        self.add_field(
            'denoiser_n_tiles',
            'Denoiser number of tiles',
            'choice',
            choices=['1', '4', '9', '16']
        )
        self.add_field(
            'denoiser_tile_overlap',
            'Denoiser tile overlap',
            'float',
            min_value=0.0,
            max_value=0.5,
            increment=0.01
        )
        self.add_field(
            'denoiser_batch_size',
            'Denoiser batch size',
            'int',
            min_value=1,
            max_value=32,
            increment=1
        )


class RefinerSettings(SettingsPanel):
    HEADING = 'Refiner'

    def _init_fields(self):
        _, ir_toggle, _ = self.add_field(
            'use_inverse_rendering',
            'Use inverse rendering',
            'bool',
        )
        use_percept = self.add_field(
            'use_perceptual_model',
            'Use perceptual model',
            'bool',
        )
        use_latents = self.add_field(
            'use_latents_model',
            'Use latents model',
            'bool',
        )
        use_rcf = self.add_field(
            'use_rcf_model',
            'Use rcf model',
            'bool',
        )

        def toggle_inverse_rendering(event=None):
            """
            Toggle the fields that require inverse rendering.
            """
            enabled = ir_toggle.GetValue()
            for field_group in [use_percept, use_latents, use_rcf]:
                label, field, help_text = field_group
                field.Enable(enabled)
                label.SetForegroundColour('black' if enabled else 'grey')
                help_text.SetForegroundColour('black' if enabled else 'grey')
            if event is not None:
                event.Skip()

        ir_toggle.Bind(wx.EVT_CHECKBOX, toggle_inverse_rendering)
        toggle_inverse_rendering()


class RendererSettings(SettingsPanel):
    HEADING = 'Renderer'

    def _init_fields(self):
        self.add_field(
            'working_image_size',
            'Working image size',
            'int',
            min_value=32,
            max_value=1024,
        )
        self.add_field(
            'spp',
            'Samples per pixel',
            'int',
            min_value=1,
            max_value=256,
        )
        self.add_field(
            'integrator_max_depth',
            'Integrator max depth',
            'int',
            min_value=1,
            max_value=64,
        )
        self.add_field(
            'integrator_rr_depth',
            'Integrator Russian-roulette depth',
            'int',
            min_value=1,
            max_value=64,
        )


class OptimiserSettings(SettingsPanel):
    HEADING = 'Optimiser'

    def _init_fields(self):
        self.add_field(
            'seed',
            'Seed',
            'int',
        )
        self.add_field(
            'max_steps',
            'Maximum steps',
            'int',
            max_value=1000000,
        )
        self.add_field(
            'multiscale',
            'Multiscale',
            'bool',
        )
        self.add_field(
            'acc_grad_steps',
            'Accumulated gradient steps',
            'int',
            min_value=1,
            max_value=50,
            increment=1
        )
        self.add_field(
            'clip_grad_norm',
            'Clip gradient norm',
            'float',
            min_value=0.0,
            max_value=100.0,
            increment=0.1
        )
        self.add_field(
            'opt_algorithm',
            'Optimisation algorithm',
            'choice',
            choices=['adabelief', 'adam', 'sgd', 'rmsprop', 'madgrad']
        )


class OptimiserNoiseSettings(SettingsPanel):
    HEADING = 'Optimiser noise'

    def _init_fields(self):
        self.add_field(
            'image_noise_std',
            'Image noise standard deviation',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )
        self.add_field(
            'distances_noise',
            'Distances noise standard deviation',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )
        self.add_field(
            'rotation_noise',
            'Rotation noise standard deviation',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )
        self.add_field(
            'material_roughness_noise',
            'Material roughness noise standard deviation',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )
        self.add_field(
            'material_ior_noise',
            'Material IOR noise standard deviation',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )
        self.add_field(
            'radiance_noise',
            'Radiance noise standard deviation',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )


class ConjugateFaceSwitchingSettings(SettingsPanel):
    HEADING = 'Conjugate face switching'

    def _init_fields(self):
        _, cs_toggle, _ = self.add_field(
            'use_conj_switching',
            'Use conjugate face switching',
            'bool',
        )
        cs_prob_init = self.add_field(
            'conj_switch_prob_init',
            'Initial switch probability',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )
        cs_prob_min = self.add_field(
            'conj_switch_prob_min',
            'Minimum switch probability',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )
        cs_prob_max = self.add_field(
            'conj_switch_prob_max',
            'Maximum switch probability',
            'float',
            min_value=0.0,
            max_value=1.0,
            increment=0.01
        )

        def toggle_conj_switching(event=None):
            """
            Toggle the conjugate face switching fields.
            """
            enabled = cs_toggle.GetValue()
            for field_group in [cs_prob_init, cs_prob_min, cs_prob_max]:
                label, field, help_text = field_group
                field.Enable(enabled)
                label.SetForegroundColour('black' if enabled else 'grey')
                help_text.SetForegroundColour('black' if enabled else 'grey')
            if event is not None:
                event.Skip()

        cs_toggle.Bind(wx.EVT_CHECKBOX, toggle_conj_switching)
        toggle_conj_switching()


class LearningRateSettings(SettingsPanel):
    HEADING = 'Learning rates'

    def _init_fields(self):
        self.add_field(
            'lr_distances',
            'Face distances',
            'scientific',
        )
        self.add_field(
            'lr_origin',
            'Origin position',
            'scientific',
        )
        self.add_field(
            'lr_rotation',
            'Rotation (3D)',
            'scientific',
        )
        self.add_field(
            'lr_material',
            'Material properties',
            'scientific',
        )
        self.add_field(
            'lr_light',
            'Light radiance (rgb)',
            'scientific',
        )
        self.add_field(
            'lr_switches',
            'Conjugate switching probabilities',
            'scientific',
        )


class LearningRateSchedulerSettings(SettingsPanel):
    HEADING = 'Learning rate scheduler'

    def _init_fields(self):
        self.add_field(
            'lr_scheduler',
            'Learning rate scheduler',
            'choice',
            choices=['none', 'cosine', 'step', 'multistep', 'exponential', 'cyclic', 'plateau']
        )
        self.add_field(
            'lr_min',
            'Minimum learning rate',
            'scientific',
        )
        self.add_field(
            'lr_warmup',
            'Warmup learning rate',
            'scientific',
        )
        self.add_field(
            'lr_warmup_steps',
            'Number of warmup steps',
            'int',
            min_value=0,
            max_value=100,
        )
        self.add_field(
            'lr_cycle_mul',
            'Learning rate cycle multiplier',
            'float',
            min_value=0.1,
            max_value=10.0,
            increment=0.1
        )
        self.add_field(
            'lr_cycle_decay',
            'Learning rate cycle decay',
            'float',
            min_value=0.1,
            max_value=1.0,
            increment=0.1
        )
        self.add_field(
            'lr_cycle_limit',
            'Learning rate cycle limit',
            'int',
            min_value=1,
            max_value=10,
        )
        self.add_field(
            'lr_k_decay',
            'Learning rate k decay',
            'float',
            min_value=0.1,
            max_value=1.0,
            increment=0.1
        )


class LossWeightingsSettings(SettingsPanel):
    HEADING = 'Loss weightings'

    def _init_fields(self):
        self.add_field(
            'w_img_l1',
            'L1 image loss',
            'scientific',
        )
        self.add_field(
            'w_img_l2',
            'L2 image loss',
            'scientific',
        )
        self.add_field(
            'w_perceptual',
            'Perceptual loss',
            'scientific',
        )
        self.add_field(
            'w_latent',
            'Latent encoding loss',
            'scientific',
        )
        self.add_field(
            'w_rcf',
            'RCF loss',
            'scientific',
        )
        self.add_field(
            'w_overshoot',
            'Overshoot loss',
            'scientific',
        )
        self.add_field(
            'w_symmetry',
            'Symmetry loss',
            'scientific',
        )
        self.add_field(
            'w_z_pos',
            'Z-position loss',
            'scientific',
        )
        self.add_field(
            'w_rotation_xy',
            'Rotation xy loss',
            'scientific',
        )
        self.add_field(
            'w_patches',
            'Patch loss',
            'scientific',
        )
        self.add_field(
            'w_fullsize',
            'Combined losses',
            'scientific',
        )
        self.add_field(
            'w_switch_probs',
            'Conjugate face switching probabilities loss',
            'scientific',
        )
        self.add_field(
            'w_anchors',
            'Anchors loss',
            'scientific',
        )


class LossDecayFactorsSettings(SettingsPanel):
    HEADING = 'Loss decay factors'

    def _init_fields(self):
        self.add_field(
            'l_decay_l1',
            'Multiscale L1 image loss',
            'scientific',
        )
        self.add_field(
            'l_decay_l2',
            'Multiscale L2 image loss',
            'scientific',
        )
        self.add_field(
            'l_decay_perceptual',
            'Perceptual losses',
            'scientific',
        )
        self.add_field(
            'l_decay_latent',
            'Latent losses',
            'scientific',
        )
        self.add_field(
            'l_decay_rcf',
            'RCF losses',
            'scientific',
        )


class SettingsCategoryControlPanel(wx.Panel):
    """
    Category selection panel.
    """

    def __init__(self, parent: 'SettingsSplitterPanel'):
        super().__init__(parent)
        category_options = [panel.HEADING for panel in parent.panels]

        # Create the category selection combo box
        label = wx.StaticText(self, label='Category :')
        label.SetForegroundColour(CATEGORY_SELECTOR_HEADING_FONT_COLOUR)
        label.SetFont(_get_font())
        self.category_selector = wx.ListBox(
            self,
            choices=category_options,
            style=wx.LB_SINGLE | wx.LB_ALWAYS_SB
        )
        self.category_selector.SetSelection(0)
        self.category_selector.SetForegroundColour(CATEGORY_SELECTOR_FONT_COLOUR)
        self.category_selector.SetBackgroundColour(CATEGORY_SELECTOR_BACKGROUND_COLOUR)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(label, 0, wx.TOP, 7)
        sizer.Add(self.category_selector, 1, wx.EXPAND | wx.ALL, 9)
        self.SetSizer(sizer)

        # Bind the selection event to switch panels
        self.category_selector.Bind(wx.EVT_LISTBOX, self.select_category)

    def select_category(self, event: wx.CommandEvent):
        self.GetParent().show_category(event.GetSelection())


class SettingsSplitterPanel(wx.Panel):
    """
    Sets up the category selector and the settings panels.
    """

    def __init__(self, parent: 'RefinerSettingsDialog', refiner_args: RefinerArgs):
        super().__init__(parent)
        self.refiner_args = refiner_args
        self.settings_container = wx.Panel(self, style=wx.BORDER_THEME)

        # Create the panels
        self.panels = [
            InitialPredictionSettings(self.settings_container, self.refiner_args),
            DenoiserSettings(self.settings_container, self.refiner_args),
            RefinerSettings(self.settings_container, self.refiner_args),
            RendererSettings(self.settings_container, self.refiner_args),
            OptimiserSettings(self.settings_container, self.refiner_args),
            OptimiserNoiseSettings(self.settings_container, self.refiner_args),
            ConjugateFaceSwitchingSettings(self.settings_container, self.refiner_args),
            LearningRateSettings(self.settings_container, self.refiner_args),
            LearningRateSchedulerSettings(self.settings_container, self.refiner_args),
            LossWeightingsSettings(self.settings_container, self.refiner_args),
            LossDecayFactorsSettings(self.settings_container, self.refiner_args),
        ]
        self.active_panel = self.panels[0]

        # Category selection and settings container
        self.categories = SettingsCategoryControlPanel(self)

        # Layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.categories, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.settings_container, 4, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(sizer)

    def show_category(self, selection: int):
        """
        Show the selected settings category.
        """
        for p in self.panels:
            p.Hide()
        self.active_panel = self.panels[selection]
        self.active_panel.SetSize(self.settings_container.GetSize())
        self.active_panel.Show()

        # Trigger a sizer event to make sure the help texts are properly wrapped
        size_event = wx.SizeEvent(self.active_panel.GetSize(), self.active_panel.GetId())
        self.active_panel.ProcessEvent(size_event)


class RefinerSettingsDialog(wx.Dialog):
    """
    Settings dialog.
    """

    def __init__(self, parent, refiner_args: RefinerArgs, default_refiner_args: RefinerArgs):
        super().__init__(parent, title='Settings', size=(800, 500),
                         style=wx.DEFAULT_FRAME_STYLE | wx.RESIZE_BORDER)
        self.SetMinSize((640, 400))
        self.refiner_args = refiner_args
        self.default_refiner_args = default_refiner_args
        self._init_components()
        wx.CallAfter(self.panel.show_category, 0)

        # Listen for resize events
        self.Bind(wx.EVT_SIZE, self.on_resize)

    def _init_components(self):
        """
        Initialise the components.
        """
        # Control buttons
        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        reset_btn = wx.Button(self, label='Reset to defaults')
        load_btn = wx.Button(self, label='Load settings')
        save_btn = wx.Button(self, label='Save settings')
        btn_sizer.Insert(0, reset_btn, flag=wx.RIGHT, border=0)
        btn_sizer.Insert(0, load_btn, flag=wx.RIGHT, border=5)
        btn_sizer.Insert(0, save_btn, flag=wx.RIGHT, border=5)
        reset_btn.Bind(wx.EVT_BUTTON, self.on_reset)
        load_btn.Bind(wx.EVT_BUTTON, self.on_load)
        save_btn.Bind(wx.EVT_BUTTON, self.on_save)

        # Layout
        self.panel = SettingsSplitterPanel(self, self.refiner_args)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        ctrl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctrl_sizer.Add(self.panel, 1, wx.EXPAND)
        ctrl_sizer.Add(btn_sizer, 0, wx.RIGHT | wx.ALIGN_RIGHT)
        sizer.Add(ctrl_sizer, 1, wx.EXPAND | wx.BOTTOM, 10)
        self.SetSizer(sizer)
        sizer.Layout()
        self.CenterOnScreen(wx.BOTH)

    def on_resize(self, event: wx.SizeEvent):
        """
        When the dialog is resized, resize the active panel.
        """
        self.panel.active_panel.SetSize(self.panel.settings_container.GetSize())
        event.Skip()

    def on_reset(self, event: wx.CommandEvent):
        """
        Reset the settings to their default values.
        """
        self.refiner_args = self.default_refiner_args.clone()
        active_panel_idx = self.panel.categories.category_selector.GetSelection()
        self.panel.Destroy()
        self._init_components()
        self.Layout()
        self.panel.categories.category_selector.SetSelection(active_panel_idx)
        self.panel.show_category(active_panel_idx)
        event.Skip()

    def on_load(self, event: wx.CommandEvent):
        """
        Load settings from a file.
        """
        with wx.FileDialog(
                self,
                message='Load settings',
                wildcard='YAML files (*.yaml, *.yml)|*.yaml;*.yml',
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            current_args = self.refiner_args.clone()
            filepath = Path(file_dialog.GetPath())
            try:
                with open(filepath, 'r') as f:
                    args_yml = yaml.load(f, Loader=yaml.FullLoader)
                for k, v in args_yml.items():
                    if hasattr(self.refiner_args, k):
                        old_value = getattr(self.refiner_args, k)
                        v_type = type(old_value)
                        new_type = type(v)
                        if v_type is ScalarFloat:
                            v_type = float
                        if v_type is not None and v_type is not new_type:
                            v = v_type(v)
                        setattr(self.refiner_args, k, v)

                # Update the UI
                active_panel_idx = self.panel.categories.category_selector.GetSelection()
                self.panel.Destroy()
                self._init_components()
                self.Layout()
                self.panel.categories.category_selector.SetSelection(active_panel_idx)
                self.panel.show_category(active_panel_idx)

                # Show success message
                wx.MessageBox(f'Settings loaded from {filepath}', 'Success', wx.ICON_INFORMATION)
            except Exception as e:
                self.refiner_args = current_args
                wx.MessageBox(f'Error loading settings: {e}', 'Error', wx.ICON_ERROR)
                return

        event.Skip()

    def on_save(self, event: wx.CommandEvent):
        """
        Save the settings to a file.
        """
        with wx.FileDialog(
                self,
                message='Save settings',
                wildcard='YAML files (*.yaml, *.yml)|*.yaml;*.yml',
                style=wx.FD_SAVE
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            filepath = Path(file_dialog.GetPath())
            if filepath.suffix != '.yml':
                filepath = filepath.with_suffix('.yml')
            if filepath.exists():
                dlg = wx.MessageDialog(self, message='File already exists. Overwrite?', caption='CrystalSizer3D',
                                       style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
                if dlg.ShowModal() != wx.ID_YES:
                    return
            try:
                yaml = YAML()
                yaml.preserve_quotes = True
                args_dict = self.refiner_args.to_dict()

                # Update just the values present in the refiner args file
                with open(REFINER_ARGS_PATH, 'r') as f:
                    args_yml = yaml.load(f)
                for k, v in args_dict.items():
                    if k in args_yml:
                        args_yml[k] = v
                with open(filepath, 'w') as f:
                    yaml.dump(args_yml, f)
                wx.MessageBox(f'Settings saved to {filepath}', 'Success', wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f'Error saving settings: {e}', 'Error', wx.ICON_ERROR)
        event.Skip()
