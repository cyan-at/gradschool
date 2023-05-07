#!/usr/bin/env python
# coding: utf-8

import subprocess, os, sys, time
import io
import selectors
import subprocess
import sys

# triangle #s
nums = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210]

T_t = 200
i = 0

epochs = 2 # 20000
bif = 3000
state_bound_min = -0.3
state_bound_max = 0.6
M = 100

class bcolors:
  # https://godoc.org/github.com/whitedevops/colors
  DEFAULT = "\033[39m"
  BLACK = "\033[30m"
  RED = "\033[31m"
  GREEN = "\033[32m"
  YELLOW = "\033[33m"
  BLUE = "\033[34m"
  MAGENTA = "\033[35m"
  CYAN = "\033[36m"
  LGRAY = "\033[37m"
  DARKGRAY = "\033[90m"
  FAIL = "\033[91m"
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  OKBLUE = '\033[94m'
  HEADER = '\033[95m'
  LIGHTCYAN = '\033[96m'
  WHITE = "\033[97m"

  ENDC = '\033[0m'
  BOLD = '\033[1m'
  DIM = "\033[2m"
  UNDERLINE = '\033[4m'
  BLINK = "\033[5m"
  REVERSE = "\033[7m"
  HIDDEN = "\033[8m"

  BG_DEFAULT = "\033[49m"
  BG_BLACK = "\033[40m"
  BG_RED = "\033[41m"
  BG_GREEN = "\033[42m"
  BG_YELLOW = "\033[43m"
  BG_BLUE = "\033[44m"
  BG_MAGENTA = "\033[45m"
  BG_CYAN = "\033[46m"
  BG_GRAY = "\033[47m"
  BG_DKGRAY = "\033[100m"
  BG_LRED = "\033[101m"
  BG_LGREEN = "\033[102m"
  BG_LYELLOW = "\033[103m"
  BG_LBLUE = "\033[104m"
  BG_LMAGENTA = "\033[105m"
  BG_LCYAN = "\033[106m"
  BG_WHITE = "\033[107m"

def capture_subprocess_output(subprocess_args):
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = subprocess.Popen(subprocess_args,
                               bufsize=1,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    # Create callback function for process output
    buf = io.StringIO()
    def handle_output(stream, mask):
        # Because the process' output is line buffered, there's only ever one
        # line to read when this function is called
        line = stream.readline()
        buf.write(line)
        sys.stdout.write(line)

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = (return_code == 0)

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)

def train(T_0, T_t, population):
    print(bcolors.OKGREEN + "training from %d to %d, %s" % (T_0, T_t, population) + bcolors.ENDC)

    cmd = "./train.py --crystal bcc --state_bound_min %.3f --state_bound_max %.3f --bif %d --epochs %d --model_name test --train_distribution Sobol --T_0 %d --T_t %d --sigma 0.001" % (
        state_bound_min,
        state_bound_max,
        bif,
        epochs,
        T_0,
        T_t)

    cmd = cmd.split(" ")

    if (population is None):
        # cmd += " --mu_0 \"0.2, 0.2\""
        cmd.append("--mu_0")
        cmd.append("0.2,0.2")
    else:
        # cmd += " --population %s" % (population)
        cmd.append("--population")
        cmd.append(population)

    print(cmd)

    # res = subprocess.run(
    #     cmd,
    #     shell=True,
    #     stdout=subprocess.PIPE,
    #     executable="/bin/bash")

    success, output = capture_subprocess_output(cmd)

    print("SUCCESS", success)

    control_data = "./" + os.path.basename(output.split("\n")[-2].replace("CONTROL_DATA: ", ""))

    return control_data, success

def cl(T_0, T_t, control_data, population):
    print(bcolors.OKGREEN + "cl from %d to %d, %s, %s" % (T_0, T_t, control_data, population) + bcolors.ENDC)

    cmd = "./closedloop.py --crystal bcc --state_bound_min %.3f --state_bound_max %.3f --bif %d --sigma 0.001 --epochs %d --model_name test --train_distribution Sobol --T_0 %d --T_t %d --M %d --control_data %s" % (
        state_bound_min,
        state_bound_max,
        bif,
        epochs,
        T_0,
        T_t,
        M,
        control_data)

    cmd = cmd.split(" ")

    if (population is None):
        # cmd += " --mu_0 \"0.2, 0.2\""
        cmd.append("--mu_0")
        cmd.append("0.2,0.2")
    else:
        # cmd += " --population %s" % (population)
        cmd.append("--population")
        cmd.append(population)

    print(cmd)

    success, output = capture_subprocess_output(cmd)

    print("SUCCESS", success)

    population = './' + os.path.basename(output.split("\n")[-2].replace("WITH_CONTROL: ", ""))

    return population, success

population = None
while i < len(nums) and nums[i] < T_t:
    # train from nums[i] to T_t
    control_data, success = train(nums[i], T_t, population)

    if not success:
        raise Exception('bad')

    # cl from nums[i] to nums[i+1]
    population, success = cl(nums[i], min(T_t, nums[i+1]), control_data, population)

    if not success:
        raise Exception('bad')

    i += 1
    print("")
