import numpy as np
from scipy import optimize
import rerun as rr

from homotopy_utils import discretize_path, skeleton_field_dir


def h_signature_continuous(path, skeleton):
    path_discretized = discretize_path(path)
    path_deltas = np.diff(path_discretized, axis=0)
    bs = skeleton_field_dir(skeleton, path_discretized[:-1])
    I = 0
    for b_i, p_i, delta_i in zip(bs, path_discretized, path_deltas):
        dI = np.dot(b_i, delta_i)
        I += dI
    return I


def main():
    rr.init("homotopy_demo")
    rr.connect()

    rr.log("world", rr.Transform3D())

    # This defines the obstacles, which each must be closed curves in 3D (loops)
    skeleton = np.array([
        [0.5, -2.5, -3.4],
        [0.5, -2.5, 1.8],
        [0.5, 0.5, 1.8],
        [0.5, 0.5, -3.4],
        [0.5, -2.5, -3.4],
    ])
    rr.log(f'skeleton', rr.LineStrips3D(skeleton, colors=[0, 255, 0]))

    viz_test(skeleton, np.array([0.7, 0.2, 0]), np.array([0.8, 0.3, 0]))
    input("Press enter to continue...")
    viz_test(skeleton, np.array([0.5, 0.2, 0]), np.array([0.5, 0.7, 0]))
    input("Press enter to continue...")
    viz_test(skeleton, np.array([0.2, 0.5, 0]), np.array([0.5, 0.7, 0]))


def viz_test(skeleton, p2_end_test, p1_end_test):
    points1 = np.stack((np.zeros(3), p1_end_test))

    def func(p2_end_):
        points2 = np.stack((np.zeros(3), p2_end_))
        h1 = h_signature_continuous(points1, skeleton)
        h2 = h_signature_continuous(points2, skeleton)
        c = np.abs(h2 - h1)
        return c

    func(p2_end_test)
    grad = optimize.approx_fprime(p2_end_test, func)

    rr.log("grad", rr.Arrows3D(origins=p2_end_test, vectors=grad))

    points2 = np.stack((np.zeros(3), p2_end_test))
    rr.log("points1", rr.LineStrips3D(points1, colors=[255, 0, 0]))
    rr.log("points2", rr.LineStrips3D(points2, colors=[0, 0, 255]))


if __name__ == '__main__':
    main()
