# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 
 
"""
Utility functions provided by Audacity as part of the example script pipe_test.py.
https://github.com/audacity/audacity/blob/master/scripts/piped-work/pipe_test.py
"""

import os
import sys

def pipe_setup():
    global EOL
    if sys.platform == 'win32':
        print("pipeutils.py, running on windows")
        TONAME = '\\\\.\\pipe\\ToSrvPipe'
        FROMNAME = '\\\\.\\pipe\\FromSrvPipe'
        EOL = '\r\n\0'
    else:
        print("pipeutils.py, running on linux or mac")
        TONAME = '/tmp/audacity_script_pipe.to.' + str(os.getuid())
        FROMNAME = '/tmp/audacity_script_pipe.from.' + str(os.getuid())
        EOL = '\n'

    print("Write to  \"" + TONAME +"\"")
    if not os.path.exists(TONAME):
        print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
        sys.exit()

    print("Read from \"" + FROMNAME +"\"")
    if not os.path.exists(FROMNAME):
        print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
        sys.exit()

    print("-- Both pipes exist.  Good.")

    global TOFILE
    global FROMFILE
    TOFILE = open(TONAME, 'w')
    print("-- File to write to has been opened")
    FROMFILE = open(FROMNAME, 'rt')
    print("-- File to read from has now been opened too\r\n")

def send_command(command):
    """Send a single command."""
    print("Send: >>> \n"+command)
    global TOFILE
    global EOL
    TOFILE.write(command + EOL)
    TOFILE.flush()

def get_response():
    """Return the command response."""
    result = ''
    line = ''
    global FROMFILE
    while True:
        result += line
        line = FROMFILE.readline()
        if line == '\n' and len(result) > 0:
            break
    return result

def do_command(command):
    """Send one command, and return the response."""
    send_command(command)
    response = get_response()
    print("Rcvd: <<< \n" + response)
    return response
