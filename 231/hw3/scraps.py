 1/1: import scipy
 1/2: import numpy
 1/3: import matplotlib
 2/1: import scipy
 3/1: 1 ^ 3
 3/2: 1 ^ 3 ^ 3
 3/3: 3 ^ 3
 3/4: 1 ^ 0
 3/5: 5 ^ 0
 3/6: 1273 ^ 0
 4/1: 3 * (2) // 2
 5/1: import collada
 6/1: import math
 6/2: math.sqrt(4.8289**2 - 1.68**2)
 6/3: x = math.sqrt(4.8289**2 - 1.68**2)
 6/4: 11.86295 - 2 * x
 7/1: (11.86925 - 2.8085) / 2.0
10/1: import numpy as np
10/2: np.ones(2049, 2049)
10/3: np.ones(2049, 2049)
10/4: np.ones((2049, 2049))
11/1: import numpy as np
11/2: np.power(2, 16)
11/3: np.ceil(2.5)
25/1: 1.68 / 2
26/1: import django
27/1: import ics
41/1: import pyqtgraph
42/1: import pyqt5
42/2: import PyQt5
42/3: from PyQt5 import sip
44/1: from PyQt5 import sip
45/1: import pyqtgraph
46/1: import pyqtgraph
58/1: import pyqtgraph
58/2: pyqtgraph.examples.run()
58/3: pyqtgraph.examples
60/1: import pinax
60/2: from pinax import stripe
61/1:
for x in range(5):
    print(x)
62/1: from onboard.models import User
62/2: User.objects.all().delete()
62/3: User.objects.all().delete()
63/1: from onboard.models import User
63/2: User.objects.all().delete()
63/3: User.objects.all().delete()
64/1: from onboard.models import *
64/2:
    AUser.objects.all().delete()
    Consumer.objects.all().delete()
    Producer.objects.all().delete()
    User.objects.all().delete()
    BitSet.objects.all().delete()
    Slot.objects.all().delete()
    Generator.objects.all().delete()
    Spot.objects.all().delete()
    Net.objects.all().delete()
64/3: Spot.objects.all()
65/1: from onboard.models import *
65/2: BitSet.objects.filter('aa4fb582-2f66-49c2-938a-7ab68b80be09')
65/3: BitSet.objects.filter('aa4fb582-2f66-49c2-938a-7ab68b80be09').all()
65/4: BitSet.objects.all()
65/5: list(BitSet.objects.all())
65/6: BitSet.objects.filter(uuid=request.GET['uuid'])
65/7: BitSet.objects.filter(uuid=x)
65/8: x = 'aa4fb582-2f66-49c2-938a-7ab68b80be09'
65/9: BitSet.objects.filter(uuid=x)
65/10: BitSet.objects.filter(uuid=x).all()
65/11: list(BitSet.objects.filter(uuid=x).all())
65/12: y = list(BitSet.objects.filter(uuid=x).all())
65/13: y[0]
65/14: x
65/15: x == 'aa4fb582-2f66-49c2-938a-7ab68b80be09'
66/1: from django.core.mail import send_mail
66/2: from django.conf import settings
66/3:
send_mail(
...     subject='A cool subject',
...     message='A stunning message',
...     from_email=settings.EMAIL_HOST_USER,
...     recipient_list=[settings.RECIPIENT_ADDRESS])
66/4:
send_mail(
...     subject='A cool subject',
...     message='A stunning message',
...     from_email=settings.EMAIL_HOST_USER,
...     recipient_list=['cyanatg@gmail.com'])
67/1: from yamcs.client import YamcsClient
67/2: import yamcs
67/3: yamcs.__file__
67/4: yamcs.__path__
69/1: import cv2
69/2:
cv2.recoverPose(


)
93/1: import pypangolin
94/1: import pypangolin
95/1: from pydbus import SessionBus
96/1: import numpy as np
96/2: np.ones(5, 3)
96/3: np.ones((5, 3))
96/4: np.ones((5, 1))
96/5: np.pi / 3 * np.ones((5, 1))
96/6: np.linspace(np.pi / 3, 0, endpoint = False, num=10, retstep = np.pi / 10)
96/7: np.linspace(start = np.pi / 3, 0, endpoint = False, num=10, retstep = np.pi / 10)
96/8: np.linspace(start = np.pi / 3, stop = 0, endpoint = False, num=10, retstep = np.pi / 10)
96/9: np.linspace(start = np.pi / 3, stop = np.pi / 3, endpoint = False, num=10, retstep = np.pi / 10)
96/10: np.linspace(start = 0, stop = np.pi / 3, endpoint = False, num=10, retstep = np.pi / 10)
96/11: [x for x in range(10)]
96/12: np.pi / 5 + [x for x in range(10)] * np.pi / 3
96/13: np.ones((10, 1)) * np.pi / 5 + np.array([x for x in range(10)] * np.pi / 3)
96/14: np.ones((10, 1))  + np.array([x for x in range(10)] * np.pi / 3)
96/15: np.ones((10, 1))  + np.array([x for x in range(10)]) * np.pi / 3
96/16: np.array([x for x in range(10)]) * np.pi / 3
96/17: np.ones((10, 1))  + (np.array([x for x in range(10)]) * np.pi / 3).T
96/18: np.ones((10, 1))
96/19: np.ones((1, 10))  + (np.array([x for x in range(10)]) * np.pi / 3)
96/20: np.ones((1, 10)) * np.pi / 5  + (np.array([x for x in range(10)]) * np.pi / 3)
96/21: thetas = np.ones((1, 10)) * np.pi / 5  + (np.array([x for x in range(10)]) * np.pi / 3)
96/22: thetas.shape
96/23: np.cos(thetas)
96/24: np.vstack([thetas, thetas])
96/25: np.vstack([thetas, thetas]).T
97/1: import numpy as np
97/2: np.array([1.0, 2.0, 3.0])
97/3: delta = np.array([1.0, 2.0, 3.0])
97/4: np.tile(delta, (3, 1))
97/5: np.tile(delta, (4, 1))
97/6: repeated = np.tile(delta, (4, 1))
97/7: n = 10
97/8: np.array([1+x for x in range(n)])
97/9: s = np.array([1+x for x in range(n)])
97/10: repeated * s
97/11: n = 4
97/12: s = np.array([1+x for x in range(n)])
97/13: repeated * s
97/14: repeated * s.T
97/15: s
97/16: repeated
97/17: np.matrix(s)
97/18: np.matrix(s).T
97/19: repeated * np.matrix(s).T
97/20: np.matmul(repeated, np.matrix(s).T)
97/21: np.matrix(s).T
97/22: y = np.matrix(s).T
97/23: repeated
97/24: repeated
97/25: np.matrix(s).T
97/26: s
97/27: np.tile(s, (4, 1))
97/28: np.tile(s, (4, 1)).T
97/29: repeated * np.tile(s, (4, 1)).T
97/30: np.tile(s, (3, 1)).T
97/31: repeated * np.tile(s, (3, 1)).T
97/32: repeated
97/33: now = np.array([21, 30, 35])
97/34: now + repeated * np.tile(s, (3, 1)).T
97/35: s
98/1: import urdfpy
99/1: x = 'test; apple'
99/2: timestamp, data = x.split(';')
99/3: timestamp
99/4: data
99/5: d = 123
99/6: x = [4,5,6]
99/7: d + x
99/8: [d] + x
114/1: import cadquery
114/2: import cadquery as cq
114/3:
result = cq.Workplane(cq.Plane.XY()).box(4,2, 0.5).faces(">Z").workplane().rect(3.5, 1.5, forConstruction=True)\
.vertices().cboreHole(0.125, 0.25, 0.125, depth=None)
114/4: result
114/5: build_object(result)
119/1: import pytransform3d
119/2: dir(pytransform3d)
136/1: import pycam
137/1: import gtk
192/1: [x for x in range(1, 3)]
193/1: ./mary_clamp.py
193/2: a
193/3: b
193/4: run ./mary_clamp.py
193/5: a
193/6: [math.ceil(x / 2.0) * j + math.floor(x / 2.0) * (d + 2* i + a) for x in range(1, b+1)]
193/7: [math.ceil(x / 2.0) * j for x in range(1, b+1)]
193/8: [math.ceil(2.5) * j for x in range(1, b+1)]
193/9: [math.ceil(2.5) for x in range(1, b+1)]
193/10: import numpy as np
193/11: [np.ceil(2.5) for x in range(1, b+1)]
193/12: [np.ceil(2) for x in range(1, b+1)]
193/13: b
198/1: x = [1,2,3]
198/2: y = [4,5,6]
198/3: zip(x, y)
198/4:
for z in zip(x, y):
    print(z)
203/1: import cadquery
210/1: import numpy as np
210/2: np.linspace(1.0, 5.0, 1)
210/3: np.linspace(1.0, 5.0, 2)
210/4: np.linspace(1.0, 5.0, 3)
210/5: np.linspace(1.0, 5.0, 4)
210/6: np.linspace(0.0, 5.0, 4)
210/7: np.linspace(0.0, 5.0, 3)
210/8: np.linspace(0.0, 5.0, 1+2)
210/9: np.linspace(0.0, 5.0, 2+2)
210/10: np.linspace(0.0, 5.0, 3+2)
214/1: x = [1,2,3]
214/2: np.ones((5, 3))
214/3: np.ones((5, 3)) * x
214/4: y = np.linspace(0, 5, 5)
214/5: y
214/6: repeated = np.ones((5, 3)) * x
214/7: repeated[:, 0] = y
214/8: repeated
214/9: repeated.tolist()
214/10: x = "1,2,3;4,5,6"
214/11: x.split(";")
214/12: x = x.split(";")
214/13: tokens = x.split(";")
214/14: x = "1,2,3;4,5,6"
214/15: tokens = x.split(";")
214/16: tokens = [t.split(",") for t in tokens]
214/17: tokens
214/18: list(map(lambda x: float(x), t)) for t in tokens
214/19: [list(map(lambda x: float(x), t)) for t in tokens]
226/1: import wall_borg
228/1: np.random
228/2: np.random()
228/3: np.random.rand()
228/4: np.random.random_sample
228/5: np.random.random_sample()
228/6: np.random.random_sample(5)
228/7: np.random.random_sample((5, 2))
228/8: np.random.uniform(low = -15.0, high = 15.0, size = (5, 2))
239/1: ls
239/2: cd 2020-02-04-10-16-48/
239/3: ls
239/4: import cPickle as pickle
239/5: import pickle
239/6: d = pickle.load(open('./2020-02-04-10-16-48.p', 'rb'))
240/1: import pickle
240/2: pickle.load(open('2021-01-06-13-26-57.p','rb'))
240/3: x = pickle.load(open('2021-01-06-13-26-57.p','rb'))
240/4: x.keys()
240/5: x['/move_base/goal']
240/6: type(x['/move_base/goal'])
240/7: mbg = x['/move_base/goal']
240/8: mbg[0]
268/1: x = "test a"
268/2: x[:4]
268/3: x[4:]
268/4: x[4+1:]
269/1: import sympy
269/2: from sympy import *
269/3: k = Symbol('k')
269/4: p = Symbol('p')
269/5: s = Symbol('s')
269/6: h = k / (s - p)
269/7: h
269/8: h
269/9: sympy.init_printing()
269/10: h
269/11: import matplotlib.pyplot as plt
269/12: t, s = sympy.symbols('t, s')
269/13: a = sympy.symbols('a', real=True, positive=True)
269/14: f = sympy.exp(-a*t)
269/15: f
269/16: sympy.laplace_transform(f, t, s)
269/17: k, p = sympy.symbols('k, p')
269/18: k
269/19: f = k / (s - p)
269/20: f
269/21: sympy.inverse_laplace_transform(f, t, s)
269/22: sympy.inverse_laplace_transform(f, s, t)
269/23: f = k / (s + k - p)
269/24: f
269/25: sympy.inverse_laplace_transform(f, s, t)
269/26: f2 = (k / (s + k - p)) * 1 / s
269/27: s
269/28: fs
269/29: f2
269/30: sympy.inverse_laplace_transform(f2, s, t)
269/31: f2
269/32: f2.apart(s)
269/33: sympy.inverse_laplace_transform(f2.apart(s), s, t)
269/34: sympy.inverse_laplace_transform(k * sympy.Symbol('A') / ((s - p) * s), s, t)
269/35: p = sympy.symbols('p', real=True, positive=True)
269/36: sympy.inverse_laplace_transform(k * sympy.Symbol('A') / ((s - p) * s), s, t)
269/37: sympy.inverse_laplace_transform(f2, s, t)
269/38: f2 = (k / (s + k - p)) * 1 / s
269/39: k = sympy.symbols('k', real=True, positive=True)
269/40: f2 = (k / (s + k - p)) * 1 / s
269/41: sympy.inverse_laplace_transform(f2, s, t)
269/42: k
269/43: sympy.inverse_laplace_transform(k / (s - p), s, t)
269/44: sympy.inverse_laplace_transform(k / (s - p), s, t, noconds=True)
269/45: k = sympy.symbols('w', real=True, positive=True)
269/46: w = sympy.symbols('w', real=True, positive=True)
269/47: e = sympy.symbols('e', real=True, positive=True)
269/48:
f = (w**2 / (s**2 + 2 * e * w * s + w ** 2)
)
269/49: f
269/50: sympy.inverse_laplace_transform(f, s, t, noconds = True)
269/51: simplify(sympy.inverse_laplace_transform(f, s, t, noconds = True))
269/52: f
269/53: e = sympy.symbols('e', constant=True)
269/54: e
269/55: f = (w**2 / (s**2 + 2 * e * w * s + w ** 2))
269/56: simplify(sympy.inverse_laplace_transform(f, s, t, noconds = True))
269/57: e = sympy.symbols('e', real=True, positive=True)
269/58: sympy.inverse_laplace_transform(f, s, t, noconds = True)
269/59: f = (w**2 / (s**2 + 2 * e * w * s + w ** 2))
269/60: sympy.inverse_laplace_transform(f, s, t, noconds = True)
269/61:  1.667 * 10^6
269/62: x =  1.667 * 10^6
269/63: x =  1.667 * 10e6
269/64: x
269/65: ln(0.15)
269/66: ln(0.15) / -pi
269/67: import numpy as np
269/68: ln(0.15) / -np.pi
269/69: x = ln(0.15) / -np.pi
269/70: a =  1.667 * 10e6
269/71: 1 / (a) * 1 / (60 * np.sqrt(x**2 / (1 + x**2)))
269/72: 1 / (60 * np.sqrt(x**2 / (1 + x**2)))
269/73: x
269/74: type(x)
270/1: import numpy as np
270/2: np.log(0.15)
270/3: np.log(0.15) / -np.pi
270/4: x = np.log(0.15) / -np.pi
270/5: 1 / (60 * np.sqrt(x**2 / (1 + x**2)))
270/6: (1 / (60 * np.sqrt(x**2 / (1 + x**2)))) ** 2
270/7: x = (1 / (60 * np.sqrt(x**2 / (1 + x**2)))) ** 2
270/8: a = 1.667 * 10e6
270/9: 1 / a * x
270/10: np.log(0.15)
270/11: k = np.log(0.15) / (-np.pi)
270/12: sqrt(k / (1  + k))
270/13: np.sqrt(k / (1  + k))
270/14: e = np.sqrt(k / (1  + k))
270/15: 1 / (e * 60)
270/16: wn = 1 / (e * 60)
270/17: wn * 2 / (1.667 * 10e-6)
270/18: x = np.log(0.15) / -np.pi
270/19: x = 1 / (1.667*10e-6) * (1 / (60 * np.sqrt(x**2 / (1 + x**2)))) ** 2
270/20: x
270/21: wn
270/22: wn = 1 / (e * 60)
270/23: wn
270/24: wn ** 2 / 1.667 * 10e-6
270/25: wn
270/26: (wn ** 2) / (1.667 * 10e-6)
270/27: (1.8 / 120) ** 2 / (1.667*10e-6)
270/28: (1.8 / 80) ** 2 / (1.667*10e-6)
270/29: (1.8 / 80) ** 2 * 6*10e5
270/30: (1.8 / 80) ** 2 * 6*10e6
270/31: (1.8 / 80) ** 2 * 6*10e4
270/32: 1/(6*10e4)
270/33: (1.8 / 80) ** 2 / (1.667*10e-6)
270/34: 1 / 6*10e5
270/35: 1 / (6*10e5)
270/36: 1 / (6*10e4)
270/37: 1 / (1.667 * 10e-6)
270/38: 1 / (1.667 * 10e-6) - 6*10e4
270/39: 1 / (1.667 * 10e-6) - 60000
270/40: 1 / (1.667 * 10e-6) - (6*10e4)
270/41: (1 / (1.667 * 10e-6)) - (6*10e4)
270/42: (1.8 / 80) ** 2 * 6*10e4
270/43: (1.8 / 120) ** 2 * 6*10e4
270/44: e
270/45: 1 / (60 * e)
270/46: wn
270/47: wn ** 2 * 60000
270/48: wn ** 2 * 600000
270/49: k = np.log(0.10) / (-np.pi)
270/50: e = np.sqrt(k / (1  + k))
270/51: wn = 1 / (e * 60)
270/52: wn ** 2 * 600000
270/53: k
270/54: e
270/55: np.log(0.1)
270/56: np.log(0.1) / (-np.pi)
270/57: k
270/58: k = np.log(0.1) / (-np.pi)
270/59: sqrt(k / 1 + k)
270/60: np.sqrt(k / 1 + k)
270/61: k = (np.log(0.1) / (-np.pi)) ** 2
270/62: np.sqrt(k / 1 + k)
270/63: k = (np.log(0.1) / (-np.pi)) ** 2
270/64: k
270/65: k / (1 + k)
270/66: np.sqrt(k / (1 + k))
270/67: k = (np.log(0.1) / (-np.pi)) ** 2
270/68: np.sqrt(k / (1 + k))
270/69: k = (np.log(0.15) / (-np.pi)) ** 2
270/70: np.sqrt(k / (1 + k))
270/71: e = np.sqrt(k / (1 + k))
270/72: wn = 1 / (e * 60)
270/73: wn
270/74: wn ** 2 * 600000
270/75: k = (np.log(0.1) / (-np.pi)) ** 2
270/76: e = np.sqrt(k / (1 + k))
270/77: wn = 1 / (e * 60)
270/78: wn = 1 / (e * 60)
270/79: wn ** 2 * 600000
270/80: x = np.log(0.17) / -np.pi
270/81: x
270/82: np.sqrt(x**2 / (1 + x**2))
270/83: e = np.sqrt(x**2 / (1 + x**2))
270/84: np.arctan(np.sqrt(1 - e**2) / (e))
270/85: np.arctan(np.sqrt(1 - e**2) / (e)) * 180 / np.pi
270/86: np.pi * 2 / 3 * 2 / np.sqrt(3)
270/87: (2 * np.pi / 3) / ((np.sqrt(3) / 2) * 0.9)
271/1: 0.3125 - 0.0675
271/2: import numpy as np
271/3: 0.245 / np.tan(45 * np.pi / 180)
272/1: import Queue
274/1: x = [0.0, 1.0, 2.0]
274/2: import numpy as np
274/3: np.array(x)
274/4: np.cos(np.array(x))
276/1: f = open('./rp1a_robot.urdf', 'r')
276/2: f.readlines()
276/3:
from xml.dom import minidom
import urdf
277/1:
from xml.dom import minidom
import urdf
280/1: x = [1,2]
280/2: x.extend(3)
280/3: x + [3]
302/1:
importimport pyqtgraph.opengl as gl
import pyqtgraph.opengl as gl
302/2: import pyqtgraph.opengl as gl
302/3: dir(gl)
302/4: import pyqtgraph
302/5: dir(pyqtgraph)
302/6: from pyqtgraph import Vector
303/1: import nump as np
303/2: import numpy as np
303/3: np.random(3)
303/4: np.random((1, 3))
303/5: np.random.random((1, 3))
303/6: np.random.random((1, 3))[0]
314/1: import cv2
316/1: import cv2
332/1: import sympy
332/2: from sympy import *
333/1: import sympy
333/2: from sympy import *
333/3: alpha = Symbol('a')
333/4: a = Symbol('a')
333/5: n, f, c, r, x, y, a, b = Symbol('n, f, c, r, x, y, a, b')
333/6: n, f, c, r, x, y, a, b = Symbol('n f c r x y a b')
333/7: n = Symbol('n')
333/8: f = Symbol('f')
333/9: c = Symbol('c')
333/10: r = Symbol('r')
333/11: x = Symbol('x')
333/12: y = Symbol('y')
333/13: a = Symbol('a')
333/14: b = Symbol('b')
333/15: kgl = Matrix(-a, b)
333/16: kgl = Matrix([-a, b])
333/17: kgl
333/18: sympy.init_printing
333/19: sympy.init_printing()
333/20: kgl
333/21: kgl = Matrix([-a, 0; 0, -b])
333/22: kgl = Matrix([[a, 0, -(c - x), 0], [0, -b, -(r - y), 0], [0, 0, -(n + f), n * f], [0, 0, 1, 0]])
333/23: kgl
333/24: ndc = Matrix([[-2 / c, 0, 0, 1], [0, 2/r, 0, -1], [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]])
333/25: ndc
333/26: ndc.dot(kgl)
333/27: temp = ndc.dot(kgl)
333/28: temp
333/29: ndc * kgl
333/30: temp = ndc * kgl
333/31: temp.inv()
333/32: temp.T
333/33: kgl = Matrix([[a, 0, -(c - x), 0], [0, -b, -(r - y), 0], [0, 0, -(n + f), n * f], [0, 0, 1, 0]])
333/34: kgl = Matrix([[aa, 0, -(c - x), 0], [0, -b, -(r - y), 0], [0, 0, -(n + f), n * f], [0, 0, 1, 0]])
333/35: kgl = Matrix([[-a, 0, -(c - x), 0], [0, -b, -(r - y), 0], [0, 0, -(n + f), n * f], [0, 0, 1, 0]])
333/36: kgl
333/37: ndc
333/38: ndc * kgl
333/39: simplify(ndc * kgl)
333/40: ndc
333/41: ndc = Matrix([[-2 / c, 0, 0, 1], [0, -2/r, 0, -1], [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]])
333/42: ndc * kgl
333/43: ndc = Matrix([[-2 / c, 0, 0, 1], [0, 2/r, 0, -1], [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]])
333/44: ndc * kgl
333/45: t, s = sympy.symbols('t, s')
333/46: a = sympy.symbols('a', real=True, positive=True)
333/47: f = sympy.exp(-a*t)
333/48:
n = sympy.symbols('n', real=True, positive=True)
OM
333/49: n = sympy.symbols('n', real=True, positive=True)
333/50: f = =t**n * 1(t)
333/51: f = t**n * 1(t)
333/52: f = t**n * Heaviside(t)
333/53: f
333/54: sympy.laplace_transform(f, t, s)
333/55: s ** 2
333/56: s ** 2 * (s + 8) * (s ** 2 + 6 * s + 25)
333/57: simplify(s ** 2 * (s + 8) * (s ** 2 + 6 * s + 25))
333/58: eval(s ** 2 * (s + 8) * (s ** 2 + 6 * s + 25))
337/1: import numpy as np
337/2: np.arctan(0)
337/3: np.arctan(4/5)
337/4: np.arctan(4/5) * 180 / np.pi
337/5: np.arctan(4/1) * 180 / np.pi
337/6: 180 - np.arctan(4/1) * 180 / np.pi
337/7: 180 - 126.93 * 2 - 38.6 + 104.03
337/8: 180 - 126.93 * 2 - 38.6 + 104.03 + 90
333/59: Poly
333/60: Poly(s**2, s)
333/61: Poly(s**2, s).mul(Poly(s + 8, s))
333/62: f = Poly(s**2, s).mul(Poly(s + 8, s))
333/63: f.mul(Poly(s**2 + 6 * s + 25, s))
333/64: f = Poly(s**2, s).mul(Poly(s + 10, s))
333/65: f.mul(Poly(s**2 + 6 * s + 25, s))
333/66: np.sqrt(304)
333/67: np.sqrt(304) / 2
333/68: np.arctan(4/5) + 126.93 * 2  + 90
333/69: 180 - np.arctan(4/1)
333/70: a = np.arctan(4/5) + 126.93 * 2  + 90
333/71: b = 180 - np.arctan(4/1)
333/72: 180 - a + b
333/73: a = np.arctan(4/5) * 180 / np.pi + 126.93 * 2  + 90
333/74: b = 180 - np.arctan(4/1) * 180 / np.pi
333/75: 180 - a + b
333/76: f = Poly(s**2, s).mul(Poly(s + 8, s))
333/77: f = f.mul(Poly(s**2 + 6 * s + 25))
333/78: f
333/79: a = np.arctan(-4/5) * 180 / np.pi + np.arctan(-4/3) * 180 / np.pi * 2  + 90
333/80: b = 180 - np.arctan(-4/1) * 180 / np.pi
333/81: 180 - a + b
333/82: 180 - a + b - 360
333/83: a = np.arctan(-4/5) * 180 / np.pi + np.arctan(-4/3) * 180 / np.pi * 2 - 90
333/84: b = 180 - np.arctan(-4/1) * 180 / np.pi
333/85: 180 - a + b
333/86: 180 - a + b - 360
333/87: 180 - a + b - 360 - 360
333/88: a = np.arctan(-4/5) * 180 / np.pi + (180 - np.arctan(-4/3) * 180 / np.pi) * 2 - 90
333/89: 180 - a + b - 360 - 360
333/90: 180 - a + b
333/91: a = Poly(s + 2)
333/92: b = Poly(s**2 + 4*s + 68)
333/93: c = Poly(s + 8)
333/94: d = Poly(s**2 + 4 * s + 80)
333/95: e = Poly(s ** 2)
333/96: a.mul(b) / (e.mul(c)).mul(d)
333/97: 180 - (np.arctan(8.718 / 6) * 180 / np.pi + 90 + 2 * (180-np.arctan(8.718 / 2) * 180/np.pi))
333/98: 180 - (np.arctan(8.718 / 6) * 180 / np.pi + 90 + 2 * (180-np.arctan(8.718 / 2) * 180/np.pi)) + 270
333/99: 180 - (90 + 90) + (2 * (180 - np.arctan(8/2) * 180 / np.pi) + np.arctan(8/6) * 180 / np.pi)
333/100: a = Poly(s + 2)
333/101: b = Poly(s + 4)
333/102: c = Poly(s + 1)
333/103: d = Poly(s + 5)
333/104: e = Poly(s + 10)
333/105: f = Poly(s)
333/106: a.mul(b) / ((c.mul(d)).mul(e)).mul(f)
338/1: import numpy as np
338/2: np.sqrt(120)
338/3: np.sqrt(103)
338/4: np.sqrt(10 + 2*49)
338/5: 10 + 2 * 49
338/6: 10 + 2 * 81
338/7:
np.sqrt(172
)
338/8: 26 + 64
338/9: np.sqrt(90)
339/1: import numpy as np
339/2: A = np.zeros((4, 4))
339/3: b = np.zeros((4, 6))
339/4: A = np.array([2, 1, 1, 0, 4, 3, 3, 1, 8, 7, 9, 5, 6, 7, 9, 8]).reshape((4, 4))
339/5: A
339/6: b_str = "3.0    0.0 1.0 0.9 2.1 3.141592653589793 6.0   -2.0    1.0 10.4    -491.2  -4.712388980384690 10.0 2.0 0.0 -20.2   0.12    2.243994752564138 1.0   10.0    -5.0    -5.12   -51.3   2.356194490192345"
339/7: b
339/8: b_str
339/9: b_str.replace('\t', ' ')
339/10: b_str.replace('\t', ' ').split(" ")
339/11: [float(x) for x in b_str.replace('\t', ' ').split(" ")]
339/12: b = np.array([float(x) for x in b_str.replace('\t', ' ').split(" ")]).reshape((4, 6))
339/13: b
339/14: np.linalg.solve(A, b)
339/15: A
339/16: b
339/17: np.linalg.solve(A, b[:, 0])
339/18:
def pivot(A):
    """
    Pivots matrix A - finds row with maximum first entry
    and if nessecary swaps it with the first row.
    Input Arguments
    ---------------
    Augmented Matrix A

    Returns
    -------
    Pivoted Augmented Matrix A
    """
    
    B = numpy.zeros((1,2))
    B[0,0]=A.shape[0]       
    B[0,1]=A.shape[1]
    nrows =B[0,0] #This stores dimensions of the 
    ncols =B[0,1] #matrix in an array
    
    pivot_size = numpy.abs(A[0,0])        
    #checks for largest first entry and swaps
    pivot_row = 0; 
    for i in range(int(0),int(nrows)):
        if numpy.abs(A[i,0])>pivot_size:
            pivot_size=numpy.abs(A[i,0])
            pivot_row = i
        
    if pivot_row>0:
        tmp = numpy.empty(ncols)
        tmp[:] = A[0,:]
        A[0,:] = A[pivot_row,:]
        A[pivot_row,:] = tmp[:]
        
    return A
339/19:
def elim(A):
    """
    elim(A)uses row operations to introduce
    zeros below the diagonal in first column
    of matrix A
    Input Argument
    ---------------
    Augmented Matrix A

    Returns
    -------
    A with eliminated first column
    """
    A = pivot(A)
    B = numpy.zeros((1,2))
    B[0,0]=A.shape[0]       
    B[0,1]=A.shape[1]
    nrows =B[0,0]
    ncols =B[0,1]
    
    #row operations
    rpiv = 1./float(A[0][0])
    
    for irow in range(int(1),int(nrows)):
        s=A[irow,0]*rpiv
        for jcol in range(int(0),int(ncols)):
            A[irow,jcol] = A[irow,jcol] - s*A[0,jcol]
        
    return A
339/20:
def backsub(A):
    """
    backsub(A) solves the upper triangular system

    Input Argument
    ---------------
    Augmented Matrix A

    Returns
    -------
    vector b, solution to the linear system
    """
    B = numpy.zeros((1,2))
    B[0,0]=A.shape[0]       
    B[0,1]=A.shape[1]
    n =B[0,0]
    ncols =B[0,1]
        
    x=numpy.zeros((n,1))

    x[n-1]=A[n-1,n]/A[n-1,n-1]
    for i in range(int(n-1),int(-1),int(-1)):
        for j in range(int(i+1),int(n)):
            A[i,n] = A[i,n]-A[i,j]*x[j]

        x[i] = A[i,n]/A[i,i]
    return x
339/21:
def gaussfe(A):
    """
    gaussfe(A)uses row operations to reduce A
    to upper triangular form by calling elim and
    pivot
    Input Argument
    ---------------
    Augmented Matrix A

    Returns
    -------
    A in upper triangular form
    """
    B = numpy.zeros((1,2))
    B[0,0]=A.shape[0]       
    B[0,1]=A.shape[1]
    nrows =B[0,0]
    ncols =B[0,1]

    for i in range(int(0),int(nrows-1)):
        A[i:nrows,i:ncols]=pivot(A[i:nrows,i:ncols])
        A[i:nrows,i:ncols]=elim(A[i:nrows,i:ncols])

    return A
339/22: gaussfe(A)
339/23: import numpy
339/24: gaussfe(A)
339/25: A
339/26: history
349/1: import sympy
349/2: sympy.init_printing
349/3: sympy.init_printing()
349/4: from sympy import *
349/5: t, s = sympy.symbols('t, s')
349/6: a = sympy.symbols('a', real=True, positive=True)
349/7: f = sympy.exp(-2*t)
349/8: f
349/9: f = sympy.exp(t * -2*t)
349/10: f
349/11: f
349/12: f = sympy.exp(t * -2*t)
349/13: f
349/14: f = t * sympy.exp(-2*t)
349/15: f
349/16: sympy.laplace_transform(f, t, s)
349/17: sympy.integrate(f * sympy.exp(-s * t), (t, 0, sympy.oo))
350/1: import numpy as np
350/2: np.e
351/1: import numpy as np
351/2: ((1.8 / 0.3) ** 2 ) / 2
351/3: np.sqrt(2*17)
351/4: 1.8 / np.sqrt(2*17)
351/5: 1.8 / np.sqrt(2*20)
351/6:
import os
import matplotlib.pyplot as plt   # MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
361/1: import numpy as np
361/2: np.arctan2(1, 1)
361/3: np.arctan2(1, 1) * 180 / np.pi
361/4: np.arctan2(-1, 1) * 180 / np.pi
361/5: 270 + 45 - 225
361/6: 130 / 8
361/7: import sympy
361/8: from sympy import *
361/9: sympy.init_printing()
361/10: x=Symbol('x')
361/11: x**4
361/12: x**4 + 4*s**3+6*x**2+4*x+5
361/13: x**4 + 4*s**3+6*x**2+4*x+5
361/14: x**4 + 4*x**3+6*x**2+4*x+5
361/15: f = x**4 + 4*x**3+6*x**2+4*x+5
361/16: sympy.solvee(f = 0)
361/17: sympy.solve(f = 0)
361/18: f = x**4 + 4*x**3+6*x**2+4*x+5
361/19: sympy.roots(f)
361/20: from sympy.solvers import solve
361/21: solve(f, x)
361/22: f = 4*x**3 + 12*x**2 + 128x+4
361/23: f = 4*x**3 + 12*x**2 + 128*x+4
361/24: f
361/25: f = 4*x**3 + 12*x**2 + 12*x+4
361/26: f
361/27: solve(f, x)
362/1: import numpy as np
362/2: np.ln(0.16) / -np.pi
362/3: np.log(0.16) / -np.pi
362/4: x = np.log(0.16) / -np.pi
362/5: np.sqrt(x**2 / (1 + x**2))
362/6: (2 * np.pi / 3) / (np.sqrt(3) / 2 * 0.4)
362/7: wn = (2 * np.pi / 3) / (np.sqrt(3) / 2 * 0.4)
362/8: e = np.sqrt(x**2 / (1 + x**2))
362/9: wd = wn * np.sqrt(1-e**2)
362/10: wd
362/11: np.arctan(5.22 / (10 - 3.023))
362/12: np.arctan(5.22 / (10 - 3.023)) * 180 / np.pi
362/13: np.arctan2(5.22 / (10 - 3.023)) * 180 / np.pi
362/14: np.arctan2(5.22, (10 - 3.023)) * 180 / np.pi
362/15: np.arctan2(5.22, 2.023) * 180 / np.pi
362/16: t1 = np.arctan2(5.22, (10 - 3.023)) * 180 / np.pi
362/17: t2 = np.arctan2(5.22, 2.023) * 180 / np.pi
362/18: t2 = 180 - np.arctan2(5.22, 2.023) * 180 / np.pi
362/19: t3 = 180 - np.arctan2(5.22, 3.023) * 180 / np.pi
362/20: t1 + t2 + t3
362/21: 180 - t1 + t2 + t3
362/22: 180 - (t1 + t2 + t3)
362/23: 59.9 + 43.97
362/24: np.arctan(5.235 / 3.023)
362/25: np.arctan(5.235 / 3.023) * 180 / np.pi
362/26: (180 - np.arctan(5.235 / 3.023) * 180 / np.pi) / 2
362/27: 60 -44.03
362/28: tan(15.967 / 180.0 * np.pi)
362/29: np.tan(15.967 / 180.0 * np.pi)
362/30: 5.22 / np.tan(15.967 / 180.0 * np.pi)
362/31: 5.22 / np.tan(104.03 / 180.0 * np.pi)
362/32: 3.023 + 5.22 / np.tan(104.03 / 180.0 * np.pi)
362/33:  s = -3.023 + 5.22j
362/34: s
362/35: (s+1.72) / (s+21.63)
362/36: (s+1.72) / (s+21.63) * 10 / (s * (s + 1) * (s + 10))
362/37: np.linalg.norm((s+1.72) / (s+21.63) * 10 / (s * (s + 1) * (s + 10)))
362/38: 1 / np.linalg.norm((s+1.72) / (s+21.63) * 10 / (s * (s + 1) * (s + 10)))
362/39: -3 / 50
362/40: 0.06 / 50
362/41: s = -1 + j
362/42: s = -1 + 1j
362/43: 1 / (s * (s + 2)) * (s + 0.1) / (s + 0.02)
362/44: 1 / (s * (s + 2)) * (s + 0.1) / (s + 0.02)
362/45: np.linalg.norm(1 / (s * (s + 2)) * (s + 0.1) / (s + 0.02))
362/46: 1 / np.linalg.norm(1 / (s * (s + 2)) * (s + 0.1) / (s + 0.02))
363/1: import numpy as np
363/2: np.linalg.cholesky()
363/3:
a = np.array([[0.00 , 1.857]
[0.05 , 1.597]
[0.10 , 1.374]
[0.15 , 1.273]
[0.20 , 1.157]
[0.25  ,1.114 ]
[0.30 , 1.131]
[0.35 , 1.217]
[0.40 , 1.099]
[0.45 , 1.102]
[0.50 , 1.132]
[0.55  ,1.085 ]
[0.60  ,1.134 ]
[0.65  ,1.191 ]
[0.70  ,1.176 ]
[0.75  ,1.212 ]
[0.80  ,1.121 ]
[0.85  ,1.144 ]
[0.90  ,0.934 ]
[0.95 , 0.876]
[1.00 , 0.486]])
363/4:
a = [[0.00 , 1.857]
[0.05 , 1.597]
[0.10 , 1.374]
[0.15 , 1.273]
[0.20 , 1.157]
[0.25  ,1.114 ]
[0.30 , 1.131]
[0.35 , 1.217]
[0.40 , 1.099]
[0.45 , 1.102]
[0.50 , 1.132]
[0.55  ,1.085 ]
[0.60  ,1.134 ]
[0.65  ,1.191 ]
[0.70  ,1.176 ]
[0.75  ,1.212 ]
[0.80  ,1.121 ]
[0.85  ,1.144 ]
[0.90  ,0.934 ]
[0.95 , 0.876]
[1.00 , 0.486]]
363/5:
a = [[0.00, 1.857]
[0.05, 1.597]
[0.10, 1.374]
[0.15, 1.273]
[0.20, 1.157]
[0.25 ,1.114 ]
[0.30, 1.131]
[0.35, 1.217]
[0.40, 1.099]
[0.45, 1.102]
[0.50, 1.132]
[0.55,1.085 ]
[0.60,1.134 ]
[0.65,1.191 ]
[0.70,1.176 ]
[0.75,1.212 ]
[0.80,1.121 ]
[0.85,1.144 ]
[0.90,0.934 ]
[0.95, 0.876]
[1.00, 0.486]]
363/6:
a = [[0.00, 1.857],
[0.05, 1.597],
[0.10, 1.374],
[0.15, 1.273],
[0.20, 1.157],
[0.25 ,1.114 ],
[0.30, 1.131],
[0.35, 1.217],
[0.40, 1.099],
[0.45, 1.102],
[0.50, 1.132],
[0.55,1.085 ],
[0.60,1.134 ],
[0.65,1.191 ],
[0.70,1.176 ],
[0.75,1.212 ],
[0.80,1.121 ],
[0.85,1.144 ],
[0.90,0.934 ],
[0.95, 0.876],
[1.00, 0.486]]
363/7: a = np.array(a)
363/8: np.linalg.cholesky(a)
363/9: np.linalg.cholesky(a.T * a)
363/10: a.T
363/11: np.dot(a.T, a)
363/12: np.linalg.cholesky(np.dot(a.T, a))
363/13: np.linalg.cholesky(np.dot(a.T, a))
363/14: a
363/15: a.T
363/16: a.T[0, :]
363/17: np.dot(a.T[0, :], a[:, 0])
363/18: sum = 0.0
363/19:
for k in range(21):
    sum + a[k, 0] * a[k, 0]
363/20: sum
363/21: a[1, 0]
363/22:
for k in range(5):
    print(k)
363/23:
for k in range(5):
    print(a[k, 0])
363/24:
for k in range(21):
    sum = sum + a[k, 0] * a[k, 0]
363/25: sum
363/26: a
363/27: a2 = np.ones((21, 2))
363/28: a2[:, 1] = a[:, 0]
363/29: a2
363/30: np.dot(a2.T, a)
363/31: a
363/32: a2
363/33: np.dot(a2.T, a)
363/34: np.dot(a2.T, a2)
363/35: a2_normalized = np.dot(a2.T, a2)
363/36: np.linalg.cholesky(a2_normalized)
363/37: np.sqrt(21)
363/38: 1.387**2
363/39: a2_normalized
363/40: 10.5**2 + 7.175**2
363/41: 1.387**2
363/42: 7.175 - 1.387**2
363/43: sqrt(7.175 - 1.387**2)
363/44: np.sqrt(7.175 - 1.387**2)
363/45: np.linalg.solve(np.linalg.cholesky(a2_normalized), np.dot(a2.T, a[:, 1]))
363/46: b_normalized = np.dot(a2.T, a[:, 1])
363/47: b_normalized
363/48: a2_normalized
363/49: np.linalg.solve(a2_normalized, b_normalized)
363/50: x = np.linalg.solve(a2_normalized, b_normalized)
363/51: np.dot(a2_normalized, x)
363/52: b_normalized
363/53: a2_normalized
363/54: x
363/55: a
363/56: a2
363/57: np.dot(a2, x)
363/58: a[:, 1] - np.dot(a2, x)
363/59: err = a[:, 1] - np.dot(a2, x)
363/60: np.dot(err, err)
363/61: a2_normalized
363/62: a2_normalized[:, 0]
363/63: np.linalg.norm(a2_normalized[:, 0])
363/64: a2
363/65: np.linalg.qr(a2)
363/66: np.linalg.qr(a2, 'complete')
363/67: np.linalg.qr(a2)
363/68: q, r = np.linalg.qr(a2)
363/69: r
363/70: q
363/71: a2
363/72: a2
363/73: q, r = np.linalg.qr(a2)
363/74: r
363/75: q, r = np.linalg.qr(a2, 'complete')
363/76: q
363/77: q.shape
363/78: r
363/79: q.shape
363/80: q
363/81: q, r = np.linalg.qr(a2)
363/82: np.dot(q, r)
363/83: a2
363/84: b
363/85: a
363/86: a
363/87: a2
363/88: np.dot(a2, x)
363/89: x
363/90: a[:, 1] = np.dot(a2, x)
363/91: history
363/92:
a = [[0.00, 1.857]
[0.05, 1.597]
[0.10, 1.374]
[0.15, 1.273]
[0.20, 1.157]
[0.25 ,1.114 ]
[0.30, 1.131]
[0.35, 1.217]
[0.40, 1.099]
[0.45, 1.102]
[0.50, 1.132]
[0.55,1.085 ]
[0.60,1.134 ]
[0.65,1.191 ]
[0.70,1.176 ]
[0.75,1.212 ]
[0.80,1.121 ]
[0.85,1.144 ]
[0.90,0.934 ]
[0.95, 0.876]
[1.00, 0.486]]
a = [[0.00, 1.857],
[0.05, 1.597],
[0.10, 1.374],
[0.15, 1.273],
[0.20, 1.157],
[0.25 ,1.114 ],
[0.30, 1.131],
[0.35, 1.217],
[0.40, 1.099],
[0.45, 1.102],
[0.50, 1.132],
[0.55,1.085 ],
[0.60,1.134 ],
[0.65,1.191 ],
[0.70,1.176 ],
[0.75,1.212 ],
[0.80,1.121 ],
[0.85,1.144 ],
[0.90,0.934 ],
[0.95, 0.876],
[1.00, 0.486]]
a = np.array(a)
363/93:


a = [[0.00, 1.857],
[0.05, 1.597],
[0.10, 1.374],
[0.15, 1.273],
[0.20, 1.157],
[0.25 ,1.114 ],
[0.30, 1.131],
[0.35, 1.217],
[0.40, 1.099],
[0.45, 1.102],
[0.50, 1.132],
[0.55,1.085 ],
[0.60,1.134 ],
[0.65,1.191 ],
[0.70,1.176 ],
[0.75,1.212 ],
[0.80,1.121 ],
[0.85,1.144 ],
[0.90,0.934 ],
[0.95, 0.876],
[1.00, 0.486]]
a = np.array(a)
363/94: a2
363/95: b = a[:, 1]
363/96: b - np.dot(a2, x)
363/97: b
363/98: np.dot(a2, x)
363/99: np.dot(a2, x)
363/100: np.linalg.norm(b - np.dot(a2, x))
363/101: a
363/102: a2 = np.ones((21, 3))
363/103: a2[:, 1] = a[:, 0]
363/104: a2[:, 2] = a[:, 0]**2
363/105: a2
363/106: a2
363/107: b
363/108: b.T
363/109: np.dot(a2.T, a2)
363/110: np.dot(a2.T, a[:, 1])
363/111: b2 = np.dot(a2.T, a[:, 1])
363/112: a3 = np.dot(a2.T, a2)
363/113: np.linalg.solve(a3, b2)
363/114: np.power(3, 5)
365/1: d = { 1 : 2 }
365/2:
for k,v in d.items():
    print k
365/3:
for k,v in d.items():
    print(k)
366/1: import std_srvs
366/2: dir(std_srvs)
367/1: import numpy as np
367/2: 100 / (1e-2 * 1.01 * (1 + 2*1e-4))
367/3: np.log(1)
367/4: 20*log10(100 / (1e-2 * 1.01 * (1 + 2*1e-4)))
367/5: 20*np.log10(100 / (1e-2 * 1.01 * (1 + 2*1e-4)))
367/6: 20*np.log10(1 / ((1e-2 * (1e-2 + 1) * (0.02 * 0.01 + 1))))
367/7: 20*np.log10(1 / ((s * (s + 1) * (0.02 * s + 1))))
367/8: s = 1e-1
367/9: 20*np.log10(1 / ((s * (s + 1) * (0.02 * s + 1))))
367/10: s = 1
367/11: 20*np.log10(1 / ((s * (s + 1) * (0.02 * s + 1))))
367/12: s = 10
367/13: 20*np.log10(1 / ((s * (s + 1) * (0.02 * s + 1))))
367/14: s = 100
367/15: 20*np.log10(1 / ((s * (s + 1) * (0.02 * s + 1))))
367/16: s = 1000
367/17: 20*np.log10(1 / ((s * (s + 1) * (0.02 * s + 1))))
367/18: np.arctan2(0.02, 1)
368/1: import numpy as np
368/2: s = 1e-2
368/3: np.log10((s^2 + 4) / (s * (s^2 + 1)))
368/4: -20*np.log10((s**2 + 4) / (s * (s**2 + 1)))
368/5: 20*np.log10((s**2 + 4) / (s * (s**2 + 1)))
368/6: 20*np.log10(1/s)
368/7: 20*np.log10(-50)
368/8: 10^(-50/20)
368/9: 10^(-50/20.0)
368/10: 10**(-50/20.0)
368/11: 20**2.5
368/12: 10**2.5
368/13: 10**2
368/14: 10**2.1
368/15: 10**2.2
368/16: 10**2.3
368/17: 10**2.4
368/18: 10**(47.7/20)
368/19: np.tan(60 * np.pi / 180)
368/20: 10**(-28.8/20)
368/21: 1 / (10**(-28.8/20))
368/22: 1 / (10**(-28.8/20)) / 50
368/23: 1 / (10**(-28.8/20))
368/24: 27.5/25
368/25: 96/16
368/26: (180 + 360) / 4
368/27: (180 + 360 * 2) / 4
368/28: 180 - (180 + 360 * 2) / 4
368/29: np.arctan2(4.89, 1)
368/30: np.arctan2(4.89, 1) * 180 / np.pi
368/31: 180 - np.arctan2(4.89, 1) * 180 / np.pi
368/32: (180 - np.arctan2(4.89, 1) * 180 / np.pi + 180)
368/33: 180 - (180 - np.arctan2(4.89, 1) * 180 / np.pi + 180)
368/34: np.arctan2(1)
368/35: np.arctan2(1, 0)
368/36: np.arctan2(0, 0)
369/1: from serial import Serial
369/2: s = serial.Serial('/dev/ttyACM0', 115200)
369/3: s = Serial('/dev/ttyACM0', 115200)
369/4: s.open()
369/5: s.read()
369/6: s.read()
369/7: s.read()
369/8: s.read()
369/9: s.read()
369/10: s.read()
369/11: s.read()
369/12: s.read()
369/13: s.read()
369/14: s.read()
369/15: s.read()
369/16: s.read()
369/17: s.read()
369/18: s.read()
369/19: s.read()
369/20: s.read()
369/21: s.read()
369/22: s.read()
369/23: s.read()
369/24: s.read()
369/25: s.open()
369/26: s.read()
369/27: s.read()
370/1: import serial
370/2: serial.Serial
371/1: import serila
371/2: from serial import Serial
372/1: import serial
372/2: dir(serial)
373/1: import gobject
373/2: dir(gobject)
374/1: import scipy
374/2: from scipy.linalg import hessenberg
374/3: import numpy as np
374/4: a = np.array([[5, 4, 1, 1], [4, 5, 1, 1], [1, 1, 4, 2], [1, 1, 2, 4]])
374/5: a
374/6: hessenberg(a)
375/1: import numpy as np
375/2: import scipy
375/3: np.array([[3, 1, 0], [1, 2, 1], [0, 1, 1]])
375/4: x = np.array([[3, 1, 0], [1, 2, 1], [0, 1, 1]])
375/5: np.linalg.eigvals(x)
375/6: np.linalg.eig(x)
375/7: np.linalg.qr(x)
375/8: q, r = np.linalg.qr(x)
375/9: np.dot(r, q)
375/10: q
375/11: np.linalg.eig(x)
375/12: a
375/13: x
375/14: q4 = np.array([[2, 1, 3, 4], [1, -3, 1, 5], [3, 1, 6, -2], [4, 5, -2, -1]])
375/15: q4
375/16: np.linalg.inv(q4)
375/17: np.linalg.eig(q4)
375/18: np.eye(4) * -8.0286
375/19: temp = np.eye(4) * -8.0286
375/20: q4 - temp
375/21: temp2 = q4 - temp
375/22: np.linalg.inv(temp2)
375/23: np.linalg.inv(temp2)x
375/24: x
375/25: x
375/26: np.linalg.eig(x)
375/27: np.linalg.qr(x)
375/28: np.linalg.qr(x)
375/29: q, r = np.linalg.qr(x)
375/30: q
375/31: r
375/32: x
375/33: q4
375/34: q3 = q4
375/35: np.linalg.eig(q3)
375/36: eigs, _ = np.linalg.eig(q3)
375/37: eigs
375/38:
for eig in eigs:
    temp = q3 - np.eye(4) * eig
    print np.linalg.inv(temp)
375/39:
for eig in eigs:
    temp = q3 - np.eye(4) * eig
    print(np.linalg.inv(temp))
375/40: eig[0]
375/41: eigs
375/42: eigs[0]
375/43: q3 - np.eye(4) * eigs[0]
375/44: np.linalg.inv(q3 - np.eye(4) * eigs[0])
376/1: q4 = np.array([[2, 1, 3, 4], [1, -3, 1, 5], [3, 1, 6, -2], [4, 5, -2, -1]])
376/2: import numpy as np
376/3: q4 = np.array([[2, 1, 3, 4], [1, -3, 1, 5], [3, 1, 6, -2], [4, 5, -2, -1]])
376/4: q4
376/5: e, ev = np.linalg.eig(q4)
376/6: e
376/7: q4 - np.eye(4) * e[0]
376/8: temp = q4 - np.eye(4) * e[0]
376/9: np.linalg.inv(temp)
376/10: ls
376/11: q2 = np.array([[3, 1, 0], [1, 2, 1], [0, 1, 1]])
376/12: q2
376/13: e, ev = np.linalg.eig(q2)
376/14: e
376/15: ev
376/16: q3
376/17: q3 = np.array([[3, 1, 0], [1, 2, 1], [0, 1, 1]])
376/18: q, r = np.linalg.qr(q3)
376/19: q
376/20: r
376/21: import scipy
376/22: scipy.linalg.hessenberg(q3)
376/23: from scipy.linalg import hessenberg
376/24: hessenberg(q3)
376/25: q3_temp = hessenberg(q3)
376/26: q, r = np.linalg.qr(q3_temp)
376/27: q
376/28: r
376/29: np.dot(r, q)
376/30: q1 = np.array([[5, 4, 1, 1], [4, 5, 1, 1], [1, 1, 4, 2], [1, 1, 2, 4]])
376/31: hessenberg(q1)
376/32: q1.shape
376/33: q1.shape[0]
376/34: q3_temp
376/35: hessenberg(q3)
376/36: hessenberg(q3)
376/37:
x = np.array([[1  ,0 , 0],
[0  ,1 , 0],
[0  ,0 , 1],
[-1 ,1 , 0],
[-1 ,0 , 1],
[0  ,-1, 1]])
376/38: q, r = np.linalg.qr(x)
376/39: q
376/40: r
376/41:
A = np.array([
[3.0, 1.0, 0.0],
[1.0, 2.0, 1.0],
[0.0, 1.0, 1.0]])
376/42: q, r = np.linalg.qr(A)
376/43: q
376/44: r
377/1: import numpy as np
377/2: w = 2*np.pi / (3600 * 24)
377/3: o = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, -2*w, 0, 0], [-6*w^3, 0, 0, -4*w^2]])
377/4: o = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, -2*w, 0, 0], [-6*w**3, 0, 0, -4*w**2]])
377/5: o
377/6: w
377/7: np.linalg.rank(w)
377/8: np.rank(o)
377/9: np.linalg.matrix_rank(o)
377/10: c = np.array([0, 0, 1, ])
377/11: a = np.array([[0, 1, 0 , 0], [3*w**2, 0, 0, 2*w], [0, 0, 0, 1], [0, -2*w, 0, 0]])
377/12: ca
377/13: np.dot(c,a)
377/14: c = np.array([0, 0, 1, 0])
377/15: np.dot(c,a)
377/16: o = np.array([c, np.dot(c,a), np.linalg.multi_dot([c, a, a]), np.linalg.multi_dot([c, a, a, a])])
377/17: o
377/18: np.linalg.matrix_rank(o)
382/1: import symbol
382/2: from sympy import *
382/3: sympy.
382/4: sympy.init_printing()
382/5: import sympy
382/6: sympy.init_printing()
382/7: s = sympy.Symbol('s')
382/8: import numpy as np
382/9: 2 * np.pi / (3600 * 24)
382/10: W = 2 * np.pi / (3600 * 24)
382/11: w = 2 * np.pi / (3600 * 24)
382/12: m = Matrix([[s, 1, 0, 0], [3*w**2, s, 0, 2*w], [0, 0, s, 1], [0, -2*w, 0, s]])
382/13: m.inv()
382/14: w = sympy.Symbol('w')
382/15: m = Matrix([[s, 1, 0, 0], [3*w**2, s, 0, 2*w], [0, 0, s, 1], [0, -2*w, 0, s]])
382/16: m.inv()
382/17: m_inv = m.inv()
382/18: m_inv[:, -1]
382/19: b = Matrix([[0], [0], [0], [1]])
382/20: c = Matrix([0, 0, 1, 0])
382/21: m_inv * b
382/22: m_inv
382/23: m_inv * b
382/24: c * m_inv * b
382/25: m_inv * b
382/26: temp = m_inv * b
382/27: c * temp
382/28: c
382/29: c = Matrix([[0, 0, 1, 0]])
382/30: c
382/31: c * temp
382/32: l = Symbol('l')
382/33: m
382/34: m[0, 2]
382/35: l1, l2, l3, l4 = Symbol('l1 l2 l3 l4')
382/36: l1, l2, l3, l4 = Symbol('l1, l2, l3, l4')
382/37: l1 = Symbol('l1')
382/38: l2 = Symbol('l2')
382/39: l3 = Symbol('l3')
382/40: l4 = Symbol('l4')
382/41: m[0, 2] = m[0, 2] + l1
382/42: m
382/43: m[1, 2] = m[1, 2] + l2
382/44: m[2, 2] = m[2, 2] + l3
382/45: m[3, 2] = m[3, 2] + l4
382/46: m
382/47: det(m)
382/48: a = (s + 2w)
382/49: a = (s + 2*w)
382/50: b = (s + 3*w)
382/51: c = (s**2 + 6*w*s+18*w**2)
382/52: a * b * c
382/53: simplify(a * b * c)
382/54: eval(a * b * c)
382/55: expand(a * b * c)
382/56: w
382/57: we = 2 * np.pi / (3600 * 24)
382/58: we
382/59: l3 = 11*we
382/60: l3
382/61: l2 = 11*w**3
382/62: l2 = 11*we**3
382/63: l2
382/64: l2 = (11*we**3 - 126*we**3) / (2*we)
382/65: l2
382/66: l4 = (we**2 - 54*we**2)
382/67: l4
382/68: l1 = (108*we**4 - 3*l4*we**2) / (6*w**3)
382/69: l1
382/70: l1 = (108*we**4 - 3*l4*we**2) / (6*we**3)
382/71: l1
383/1: import sympy
383/2: from sympy import *
383/3: a = Symbol('a')
383/4: b = Symbol('b')
383/5: m = Matrix([[0, 1], [-a, -b]])
383/6: sympy.init_printing()
383/7: m
383/8: m.eigenvals()
383/9: s = Symbol('s')
383/10: k = Symbol('k')
383/11: g = 1 / (s**2 + s)
383/12: g
383/13: d = k * ((s + 1/8)**2) / (s**2 + 1)
383/14: d
383/15: (d * g) / (1 + d * g)
383/16: t = (d * g) / (1 + d * g)
383/17: expand(t)
383/18: (s**2 + s) * (s + 1) + k * (s + 1/8)**2
383/19: g = (s**2 + s) * (s + 1) + k * (s + 1/8)**2
383/20: expand(g)
383/21: g = (s**2 + s) * (s**2 + 1) + k * (s + 1/8)**2
383/22: g
383/23: expand(g)
383/24: 1/64
383/25: 1 / 0.8775
383/26: 17 + 63
384/1: import numpy as np
384/2: np.array([0, 0, 1/np.sqrt(2)])
384/3: v = np.array([0, 0, 1/np.sqrt(2)])
384/4: v
384/5: v.T
384/6: a = np.array([2,3,4])
384/7: np.dot(v, a)
384/8: v * np.dot(v, a)
384/9: 2 / (np.dot(v, v)) * v * np.dot(v, a)
384/10: a - 2 / (np.dot(v, v)) * v * np.dot(v, a)
384/11: v
384/12: v = np.array([0, 0, 1/2])
384/13: a - 2 / (np.dot(v, v)) * v * np.dot(v, a)
384/14: v = np.array([0, 0, 1])
384/15: a - 2 / (np.dot(v, v)) * v * np.dot(v, a)
384/16: v = np.array([0, 0, 4])
384/17: a - 2 / (np.dot(v, v)) * v * np.dot(v, a)
384/18: v = np.array([0, 0, 1/(2*sqrt(2))])
384/19: v = np.array([0, 0, 1/(2*np.sqrt(2))])
384/20: a - 2 / (np.dot(v, v)) * v * np.dot(v, a)
384/21: v = np.array([0, 0, 1])
384/22: a - 2 / (np.dot(v, v)) * v * np.dot(v, a)
384/23: v = np.array([0, 0, 1/np.sqrt(2)])
384/24: a - 2 * v * np.dot(v, a)
393/1: import import pyftdi.serialext
393/2: import pyftdi.serialext
393/3: import pyftdi
394/1: import pyftdi
394/2: dir(pyftdi)
395/1: import pyftdi
395/2: dir(pyftdi)
396/1: from pyftdi.ftdi import Ftdi
396/2: dir(Ftdi)
396/3: import pyftdi.serialext
396/4: port = pyftdi.serialext.serial_for_url('ftdi://ftdi:2232h/2', baudrate=3000000)
396/5: Ftdi
396/6: Ftdi.show_devices()
397/1: import pyftdi
398/1: import pyftdi.serialext
398/2: import pyftdi.serialext
398/3: dir(serialext)
399/1: x = [1,2,3]
399/2: x.extend([4,5])
399/3: x
399/4: y = x + [6, 7]
399/5: y
399/6: x = "- git: {local-name: cfs/cfe, uri: 'ssh://git@trunk.arc.nasa.gov:7999/vipersw/cfs_cfe.git'}"
399/7: import re
399/8: re.compile("{.*}")
399/9: re.match(x)
399/10: temp = re.compile("{.*}")
399/11: temp.match(x)
399/12: temp
399/13: y = temp.match(x)
399/14: y
399/15:
if y:
    print("hi")
399/16: temp = re.compile("/{.*}/g")
399/17: y = temp.match(x)
399/18: y
399/19: x
399/20: temp = re.compile("{.*}")
399/21: temp.match(x)
399/22: temp = re.compile("a")
399/23: temp.match("a")
399/24: temp = re.compile("\{")
399/25: temp.match("{")
399/26: temp = re.compile("\{.*\}")
399/27: temp.match(x)
399/28: temp = re.compile("\{\.*\}")
399/29: temp.match(x)
399/30: temp = re.compile("\.")
399/31: temp.match(x)
399/32: temp.match(".")
399/33: temp = re.compile(".*")
399/34: temp.match(".")
399/35: temp.match("afasdf")
399/36: temp = re.compile("\{.*\}")
399/37: temp.match("afasdf")
399/38: temp.match("{afasdf}")
399/39: x
399/40: temp.match(x)
399/41: temp = re.compile(".*\{.*\}")
399/42: temp.match(x)
399/43: temp = re.compile("\{.*\}")
399/44: temp.search(x)
399/45: m = temp.search(x)
399/46: m.group(0)
399/47: import json
399/48: json.loads(m.group(0))
399/49: m.group(0)
399/50: type(m.group(0))
399/51: import yaml
399/52: s = m.group(0)
399/53: yaml.safe_load(s)
399/54: s
399/55: x
399/56: yaml.safe_load(x)
399/57: t = yaml.safe_load(x)
399/58: t[0]
399/59: t[0]['git']
399/60: t[0]['git']['local-name']
400/1: import yaml
401/1: x = [1,2,3]
401/2: x.extend([1,2,3])
401/3: x
402/1: import PyQt4
403/1: import svg
404/1: import svg
405/1: import pyqtgraph
405/2: dir(pyqtgraph)
405/3: dir(pyqtgraph.Qt)
405/4: QtCore.Qt.ItemDataRole.DisplayRole
405/5: from pyqtgraph.Qt import QtCore
405/6: QtCore.Qt.ItemDataRole.DisplayRole
405/7: dir(QtCore.Qt.ItemDataRole)
406/1: import PyQt5
406/2: dir(PyQt5)
407/1: import PyQt5
409/1: import reportlab.graphics
409/2: dir(reportlab.graphics)
409/3: :q!
410/1: import sympy
410/2: sympy.init_printing()
410/3: from sympy import *
410/4: A = Matrix('A')
410/5: A = Symbol('A')
410/6: A
410/7: t = Symbol
410/8: t = Symbol('t')
410/9: exp(A * t)
410/10: tau = Symbol('u')
410/11: B = Symbol('B')
412/1: import sympy
412/2: sympy.init_printing()
412/3: u = Symbol('u')
412/4: from sympy import *
412/5: u = Symbol('u')
412/6: (u + u**3)
412/7: a = Symbol('a')
412/8: 1 / a * (u + u**3) * (1 + 1 / a * (u + u**3) ** 2)
412/9: simplify(1 / a * (u + u**3) * (1 + 1 / a * (u + u**3) ** 2))
412/10: f = 1 / a * (u + u**3) * (1 + 1 / a * (u + u**3) ** 2)
412/11: eval(f)
412/12: evaluate(f)
412/13: expand(f)
412/14: u
412/15: v
412/16: v = Symbol('v')
412/17: v = 1 / a * (u + u**3)
412/18: v
412/19: v * (1 + v**2)
412/20: expand(v * (1 + v**2))
412/21: expand(v * (1 + v**2)) / u
412/22: expand(v * (1 + v**2) / u)
420/1: clear
420/2: from sympy import *
420/3: import sympy
420/4: sympy.init_printing()
420/5: x1 = Symbol('x1')
420/6: x2 = Symbol('x2')
420/7: x = Function('x')(t)
420/8: t = symbols('t')
420/9: x = Function('x')(t)
420/10: x
420/11: x1, x2, t = symbols('x1 x2 t')
420/12: x1 = Function('x1')(t)
420/13: x2 = Function('x2')(t)
420/14: Derivative(x1, t)
420/15: g = Function('g')(t)
420/16: Eq(Derivative(x1, t), -x1 - g*x2)
420/17: Eq(Derivative(x2, t), x1 - x2)
420/18: f2 = Eq(Derivative(x2, t), x1 - x2)
420/19: f1 = Eq(Derivative(x1, t), -x1 - g*x2)
420/20: V = Function('v')(x1, x2, t)
420/21: f3 = Eq(V, x1**2 + (1 + g)*x2**2)
420/22: f3
420/23: Derivative(f3, t)
420/24: eval(Derivative(f3, t))
420/25: Derivative(x1**2 + (1 + g)*x2**2)
420/26: f4 = x1**2 + (1 + g)*x2**2
420/27: f4.diff(t)
420/28: f4.diff(x1)
420/29: f4.diff(x2)
420/30: [f4.diff(x1) f4.diff(x2)]
420/31: [f4.diff(x1), f4.diff(x2)]
420/32: f2
420/33: f5 = x1 - x2
420/34: f5
420/35: f6 = -x1 - g*x2
420/36: f6
420/37: [f4.diff(x1), f4.diff(x2)].T
420/38: transpose([f4.diff(x1), f4.diff(x2)])
420/39: V
420/40: V.rhs
420/41: V
420/42: f3
420/43: f3.rhs
420/44: f3.rhs.diff(t)
420/45: f3.rhs.diff(x1)
420/46: f3.rhs.diff(x2)
420/47: gradient = [f3.rhs.diff(x1) f3.rhs.diff(x2)]
420/48: gradient = [f3.rhs.diff(x1), f3.rhs.diff(x2)]
420/49: f5
420/50: f6
420/51: [f6, f5]
420/52: gradient*[f6, f5]
420/53: gradient.T
420/54: Matrix(gradient)
420/55: Matrix(gradient) * Matrix([f6, f5])
420/56: Matrix(gradient) * Matrix([f6, f5]).T
420/57: Matrix(gradient).T * Matrix([f6, f5])
420/58: f3.rhs.diff(t)
420/59: f3.rhs.diff(t) + Matrix(gradient).T * Matrix([f6, f5])
420/60: f3.rhs.diff(t) + Matrix(gradient).T * Matrix([f6, f5])[0]
420/61: Matrix(gradient).T * Matrix([f6, f5])
420/62: (Matrix(gradient).T * Matrix([f6, f5]))[0]
420/63: f3.rhs.diff(t) + (Matrix(gradient).T * Matrix([f6, f5]))[0]
420/64: d = f3.rhs.diff(t) + (Matrix(gradient).T * Matrix([f6, f5]))[0]
420/65: simplify(d)
420/66: sympy.init_printing()
420/67: dsolve(f2)
420/68: dsolve(f1)
   1: import sympy; sympy.init_printing()
   2: from sympy import *
   3: x1, x2, y = Symbols('x1 x2 y');
   4: x1, x2, y = Symbol('x1 x2 y');
   5: x1 = Symbol('x1')
   6: x2 = Symbol('x2')
   7: y = Symbol('y')
   8: y - y^3
   9: f = y - y^3
  10: f
  11: f = y - y**3
  12: f
  13: integrate(f, (0, x1))
  14: integrate(f, (y, 0, x1))
  15: f2 = integrate(f, (y, 0, x1))
  16: f2.diff(x1)
  17: f2.diff(x2)
  18: v = x2**2 / 2 + integrate(y - y**3, (y, 0, x1))
  19: v
  20: v.diff(x2)
  21: v.diff(x1)
  22: v.diff((x1, x2))
  23: v.diff(x2)
  24: v.diff(x2) * (-x1 - x2 + x1**3) + v.diff(x1) * (x2)
  25: h = v.diff(x2) * (-x1 - x2 + x1**3) + v.diff(x1) * (x2)
  26: simplify(h)
  27: %history -g -f hw3
  28: ls
  29: %history -g -f hw3.py
