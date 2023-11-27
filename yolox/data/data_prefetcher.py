#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch_mlu


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.mlu.Stream()
        self.input_mlu = self._input_mlu_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.mlu.stream(self.stream):
            self.input_mlu()
            self.next_target = self.next_target.mlu(non_blocking=True)

    def next(self):
        torch.mlu.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.mlu.current_stream())
        self.preload()
        return input, target

    def _input_mlu_for_image(self):
        self.next_input = self.next_input.mlu(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.mlu.current_stream())
