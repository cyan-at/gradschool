from pysketcher import *
import numpy as np
import argparse

def set_dashed_thin_blackline(*objects):
    """Set linestyle of objects to dashed, black, width=1."""
    for obj in objects:
        obj.set_linestyle('dashed')
        obj.set_linecolor('black')
        obj.set_linewidth(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, required=True, help='')
    args = parser.parse_args()

    H = 7.
    W = 6.

    drawing_tool.set_coordinate_system(xmin=0, xmax=W,
                                       ymin=0, ymax=H,
                                       axis=False)
    #drawing_tool.set_grid(True)
    drawing_tool.set_linecolor('blue')

    L = 5*H/7          # length
    P = (W/6, 0.85*H)  # rotation point
    a = 40             # angle

    gravity = Gravity(start=P+point(0.8*L,0), length=L/3)

    vertical = Line(
        P,
        P-point(0,L))
    path = Arc(
        P,
        L,
        -90,
        a)
    angle = Arc_wText(
        r'$\theta_{1}$',
        P,
        L/4, # where to show the text along radius
        -90,
        a,
        text_spacing=1/20.)
    angle.set_linecolor('black')

    rod = Line(P, path.geometric_features()['end'])

    # or shorter (and more reliable)
    mass_pt = path.geometric_features()['end']
    mass = Circle(center=mass_pt, radius=L/20.)
    mass.set_filled_curves(color='gray')
    mass.set_linecolor(color='gray')
    rod_vec = rod.geometric_features()['end'] - \
              rod.geometric_features()['start']
    unit_rod_vec = unit_vec(rod_vec)
    # mass_symbol = Text('$m_{1}$', mass_pt + L/10*unit_rod_vec)
    mass_symbol = Text('$m_{1}$', mass_pt + point(-1/2, 0))

    joint_ref_line = Line(
        mass_pt,
        mass_pt + L/4*unit_rod_vec)
    joint_ref_line.set_linecolor('black')
    joint_ref_line.set_linestyle('dashed')
    joint_ref_line.set_linewidth(1)

    theta2 = 110.0
    angle2 = Arc_wText(
        r'$\theta_{2}$',
        mass_pt,
        L/12, # where to show the text along radius
        -90 + a,
        theta2,
        text_spacing=1/20.)
    angle2.set_linecolor('black')
    angle2.set_linewidth(1)

    l2 = 1.2
    delta2 = l2*point(
        cos(radians(a + theta2 - 90)),
        sin(radians(a + theta2 - 90))
    )
    link1 = Line(
        mass_pt,
        mass_pt + delta2
        )
    link1.set_linecolor(color='gray')
    mass1_pt = link1.geometric_features()['end']

    delta2_text = mass_pt + delta2 / 2 + point(0, 0.18)
    length2 = Text('$l_{2}$', delta2_text)

    mass1 = Circle(center=mass1_pt, radius=L/20.)
    mass1.set_linecolor(color='gray')
    mass1.set_filled_curves(color='gray')
    mass1_symbol = Text('$m_{2}$', mass1_pt + point(1/2, 0))

    theta3 = 110.0
    angle3 = Arc_wText(
        r'$\theta_{3}$',
        mass_pt,
        L/10, # where to show the text along radius
        -90 + a, # starting angle w.r.t x-axis
        -theta3, # delta w.r.t starting angle
        text_spacing=1/20.)
    angle3.set_linecolor('black')
    angle3.set_linewidth(1)

    l3 = 1.5
    delta3 = l3*point(
        cos(radians(a - theta3 - 90)),
        sin(radians(a - theta3 - 90))
    )
    link2 = Line(
        mass_pt,
        mass_pt + delta3
        )
    link2.set_linecolor(color='gray')
    mass2_pt = link2.geometric_features()['end']

    delta3_text = mass_pt + delta3 / 2 + point(-.18, 0.0)
    length3 = Text('$l_{3}$', delta3_text)

    mass2 = Circle(center=mass2_pt, radius=L/20.)
    mass2.set_linecolor(color='gray')
    mass2.set_filled_curves(color='gray')
    mass2_symbol = Text('$m_{3}$', mass2_pt + point(-1/2, 0))

    # length = Distance_wText(P, mass_pt, '$l_{1}$', linestyle='solid')
    length = Text('$l_{1}$',
        P + rod_vec / 2 + point(0.18, 0.0))


    # Displace length indication
    # length.translate(L/15*point(cos(radians(a)), sin(radians(a))))


    set_dashed_thin_blackline(vertical, path)

    fig = Composition(
        {
            'g': gravity,

            'vertical': vertical,
            'path': path,
            'theta': angle,

            'rod': rod,
            'l': length, 

            'body': mass,
            'm': mass_symbol,

            'joint_ref' : joint_ref_line,
            'angle2' : angle2,

            'length2' : length2,
            'link1' : link1,
            'mass1' : mass1,
            'mass1_symbol' : mass1_symbol,

            'angle3' : angle3,
            'length3' : length3,
            'link2' : link2,
            'mass2' : mass2,
            'mass2_symbol' : mass2_symbol,
        })

    fig.draw()

    drawing_tool.display()
    drawing_tool.savefig(args.name)

    input()
