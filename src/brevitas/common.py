# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod

from brevitas import config


class ExportMixin(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._export_mode = False
        self.export_debug_name = None
        self.export_handler = None
        self.export_input_debug = False
        self.export_output_debug = False

    @property
    @abstractmethod
    def requires_export_handler(self):
        pass

    @property
    def export_mode(self):
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        if value and config.JIT_ENABLED:
            raise RuntimeError(
                "Export mode with BREVITAS_JIT is currently not supported. Save the model' "
                "state_dict to a .pth, load it back with BREVITAS_JIT=0, and call export.")
        if value and self.training:
            raise RuntimeError("Can't enter export mode during training, only during inference")
        if value and self.requires_export_handler and self.export_handler is None:
            raise RuntimeError("Can't enable export mode on a layer without an export handler")
        elif value and not self.requires_export_handler and self.export_handler is None:
            return  # don't set export mode when it's not required and there is no handler
        elif value and not self._export_mode and self.export_handler is not None:
            self.export_handler.prepare_for_export(self)
            self.export_handler.attach_debug_info(self)
        elif not value and self.export_handler is not None:
            self.export_handler = None
        self._export_mode = value
