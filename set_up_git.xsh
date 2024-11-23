#!/usr/bin/env xonsh

import os.path
import sys

CWD = os.path.dirname(__file__)

CONFIG_FILE = os.path.join(CWD, '.git', 'config')

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        if '[include]' in f.read():
            print('Git config already set up')
            sys.exists(1)

with open(CONFIG_FILE, 'a') as f:
    f.write(
'''
[include]
path = ../.gitconfig
'''
    )
