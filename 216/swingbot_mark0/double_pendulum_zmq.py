#!/usr/bin/env python3

'''
run ./gamepad.py with a usb /js controller plugged in
reacts to key presses printing to screen
press 'mode' to stop program

external zmq controlling
gamepad_zed InitEvent;1;gamepad;gamepad;0;1
gamepad_zed CleanupEvent;1;gamepad;gamepad;0;1;0
gamepad_zed fatal
'''

import os, sys, struct, array, time
from fcntl import ioctl

from common import IterableObject, Blackboard

import zmq, signal
import numpy as np

from double_pendulum import *

# constants
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 10.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = 1.05*(L1 + L2)  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 120  # how many seconds to simulate
history_len = 500  # how many trajectory points to display
dt = 0.02
t = np.arange(0, t_stop, dt)

ALPHA = 1.
K1 = 1.0
K2 = 1.0
K3 = 1.0

class Sender(IterableObject):
  def __init__(self, args):
    super(Sender, self).__init__()

    self._context = zmq.Context()
    self._socket = self._context.socket(zmq.PUB)
    self._socket.bind('tcp://127.0.0.1:55554')

    self._topics = []

    self._args = args

    self._data = np.array([0.0, 0.0, 0.0])
    self._dt_acc = 0
    self._dt = 0
    self._last_t = None

    self._data_mutex = Lock()
    self._data_condition = Condition(self._data_mutex)
    self._new_data = False

    self._state = None
    self._pressed = False
    self.i = 0

  ##################################################

  def press(self, event):
    print("pressed")
    self._pressed = True

    # self._data_mutex.acquire()
    # self._new_data = True
    # self._data_condition.notify_all()
    # self._data_mutex.release()

  def setup(self, fig, ax):
    self._fig = fig
    self._ax = ax

    self._texts = [
        ax.text(0.05, 0.9, '', transform=ax.transAxes),
        ax.text(0.05, 0.8, '', transform=ax.transAxes),
        ax.text(0.05, 0.7, '', transform=ax.transAxes),
        ax.text(0.05, 0.6, '', transform=ax.transAxes),
    ]

    # integrate your ODE using scipy.integrate.
    initial_state = np.radians([float(x) for x in self._args.initial.split(',')])

    system_params, controller_params, initial_state = deserialize(self._args)

    self.c = Acrobot(self._args,
      system_params,
      controller_params,
      initial_state,
      np.arange(0, self._args.t_stop, self._args.dt))
    self.c.init_data()

    self.c.init_plot(fig, ax, self._texts)
    self.c._data_gen_cb = self.data_gen_cb

  def do_init(self, *args, **kwargs):
    pass

  def do_cleanup(self, *args, **kwargs):
    print("teardown")

  def init_external_blackboard(self, *args):
    pass

  def data_gen_cb(self, data):
    self.i = data
    self._data = self.c.state[self.i, :]

  def do_iterate(self, *args, **kwargs):
    # print("do_iterate")

    # if file is not blocking
    # then block with a cooperative while loop
    # #cool #important #2022-01
    # this means any hanging IterateEvent threads
    # will be cooperative if prog externally killed
    while self._initialized:
      if self._last_t is not None:
        self._dt_acc += self._dt
        self._dt = time.time() - self._last_t
      self._last_t = time.time()

      # self._data[0] = (self._data[0] + 0.01) % 2*np.pi
      # self._data[0] = self._dt_acc

      target_q1_norm = np.pi / 2
      speed = 5
      freq = 5
      amp = min(target_q1_norm, self._dt_acc / speed)
      self._data[1] = amp * np.sin(self._dt_acc * freq)
      # self._data[0] = (self._data[0] + 0.05)  % (2*np.pi)
      # self._data[1] = self._dt_acc # amp * np.sin(self._data[0])
      # self._data[2] = amp * np.cos(self._data[0])

      # if self._state is not None:
      #   self._data[0] = self._state[self.i, 0]
      #   self._data[1] = self._state[self.i, 1]
      #   self._data[2] = self._state[self.i, 2]

      self.produce("hi", "test")
      time.sleep(0.01)

  ###################################

  def add_zmq_sub(self, topic_name):
    self._topics.append(topic_name)

  def produce(self, k, v):
    for topic in self._topics:
      event_name = "NewDataEvent"
      event_id = "1"
      newdata_blackboard_target = "plotter"
      ed_prefix = "datatarget"
      s = topic + " %s|%s|%s|%s|%s|%s" % (
        event_name,
        event_id,

        newdata_blackboard_target,
        # matters if this event is targeting an object
        ed_prefix,
        # only matters if the event will mess with the dispatch itself

        k,
        ",".join(["%.7f" % (x) for x in self._data]),
        )

      self._socket.send(s.encode('utf-8'))
      # print("sent %s" % (s))

  ###################################

if __name__ == '__main__':
  import argparse
  from threading import Lock, Condition, Thread

  from common import bcolors, nonempty_queue_exists
  from common import IterateEvent, CleanupEvent, InitEvent
  from common import BlackboardQueueCVED, ZmqED

  parser = argparse.ArgumentParser(
    description="")
  parser.add_argument('--autostart', type=int, default=1)

  parser.add_argument('--playback', type=int, default=1, help='')
  parser.add_argument('--history', type=int, default=500, help='')
  parser.add_argument('--dt', type=float, default=0.02, help='')
  parser.add_argument('--t_stop', type=int, default=300, help='')

  parser.add_argument('--system', type=str, default="10,1,1,1,1", help='')

  parser.add_argument('--initial', type=str, default="0,0,1,0", help='')
  '''
  q2_dot [3] cannot be 0
  http://underactuated.mit.edu/pend.html#energy_shaping
  This is true for any  theta_dot, except for theta_dot = 0
  (so it will not actually swing us up from the downright fixed point...
  but if you nudge the system just a bit, then it will start
  pumping energy and will swing all of the way up).
  '''

  parser.add_argument('--mode', type=int, default=0, help='')

  # derivs_pfl_collocated_strategy1
  parser.add_argument(
      '--control1',
      type=str,
      default="5.1,system_deserialize1,system_deserialize1") # ALPHA, K1, K2
  # derivs_pfl_collocated_taskspace
  parser.add_argument(
      '--control2',
      type=str,
      default="1.04,1.0,1.0") # K3, K4
  # derivs_pfl_collocated_energy
  parser.add_argument(
      '--control3',
      type=str,
      default="120.0,5.0,5.0") # energy_goal, K7, K8

  args = parser.parse_args()

  ############### overhead
  blackboard = {}

  # we purposely use the 'done' set to keep
  # this process alive, and the done_queue
  # being non-empty until some actor / event
  # pop's it empty and notifies 'done'
  # effectively killing the program
  blackboard["done"] = Condition()
  blackboard["done_mutex"] = Lock()
  blackboard["done_queue"] = {
    # this added == thread aware of done

    # "gamepad_run", 
    "gamepad_run_zmq",
  }

  ############### actors
  gamepad = Sender(args)
  gamepad.add_zmq_sub("datatarget")
  gamepad.init_external_blackboard(blackboard)

  blackboard["gamepad"] = gamepad

  ############### dispatches
  ed = ZmqED(
    blackboard, "gamepad", mode=1, port=55555, topics=["usb"])
  blackboard["gamepad_thread"] = Thread(
    target=ed.run,
    args=(blackboard,
      "gamepad",

      # "done",
      None,

      # bcolors.CYAN,
      None
    ))

  blackboard["gamepad_zmq_thread"] = Thread(
    target=ed.run_zmq,
    args=(blackboard,
      "gamepad",

      "done",
      # None,

      bcolors.YELLOW,
      # None
    ))

  ############### events
  blackboard["InitEvent"] = InitEvent
  blackboard["IterateEvent"] = IterateEvent
  blackboard["CleanupEvent"] = CleanupEvent

  blackboard["events"] = [InitEvent, IterateEvent, CleanupEvent]

  ############### process init

  if args.autostart:
    gamepad.init(
      ed,
      None,
      ['tag=Microsoft Corp. Xbox360 Controller'])
    if not gamepad.initialized():
      print("couldn't initialize gamepad")
      # sys.exit(1)

    blackboard["gamepad_cv"].acquire()
    blackboard["gamepad_queue"].append(
      ["IterateEvent", 1, "gamepad", "gamepad"])
    blackboard["gamepad_cv"].notify(1)
    blackboard["gamepad_cv"].release()

  ############### process lifecycle
  blackboard["gamepad_thread"].start()
  blackboard["gamepad_zmq_thread"].start()

  fig = plt.figure(figsize=(5, 4))
  fig.canvas.mpl_connect('key_press_event', gamepad.press)
  ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
  ax.set_aspect('equal')
  ax.grid()
  gamepad.setup(fig, ax)
  _ = animation.FuncAnimation(
    fig,
    gamepad.c.draw_func,
    gamepad.c.data_gen,
    blit=True,
    interval=args.dt*1000/args.playback)
  plt.title('playback speed %dx' % (args.playback))
  plt.show()

  # really it is wait on a Condition that
  # all consumers notify
  # but predicate on all queues being empty
  blackboard["done"].acquire()
  while nonempty_queue_exists(blackboard,
    [
      # queue names that are admissible
      # if they are nonempty
    ],
    verbose = False
    ):
    # ) or not blackboard["atomic_bool"]:
    blackboard["done"].wait()
  blackboard["done"].release()

  ### Shutdown procedure
  print("### SHUT DOWN ###")

  # purge all shared resources
  # set all heartbeat(hb)s to false
  for k in blackboard.keys():
    if k[-3:] != "_hb":
      continue
    print("notifying hb %s" % (k))
    blackboard[k] = False

  # notify all condition variables
  for k in blackboard.keys():
    if k[-3:] != "_cv":
      continue
    print("notifying cv %s" % (k))
    mutex_k = k[:-3] + "_mutex"
    blackboard[mutex_k].acquire()
    blackboard[k].notify_all()
    blackboard[mutex_k].release()

  # join threads
  for k in blackboard.keys():
    if k[-7:] != "_thread":
      continue
    print("joining", k)
    try:
      blackboard[k].join(5)
      if (blackboard[k]).is_alive():
        print("thread %s alive" %(k))
      else:
        print("thread %s dead" %(k))
    except Exception as e:
      print(str(e))
  print("done joining everything")

  # gamepad.cleanup(blackboard)
