#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   myargparse.py
@Time    :   2023/12/22 10:34:27
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
import argparse
from typing import Any


class MyArgumentParser(argparse.ArgumentParser):

    def __init__(self, fromfile_prefix_chars='+', **kwargs):
        super(MyArgumentParser, self).__init__(
            fromfile_prefix_chars=fromfile_prefix_chars,
            **kwargs
        )

    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split('=')
    
class MyNamespace(argparse.Namespace):

    def __getattr__(self, name: str) -> Any:
        return None