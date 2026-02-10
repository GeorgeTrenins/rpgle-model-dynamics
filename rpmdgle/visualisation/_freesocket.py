#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _socket.py
@Time    :   2024/03/25 14:21:40
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
import socket
from contextlib import closing

def find_free_port(ip=''):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((ip, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('192.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP