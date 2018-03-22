from .geom import *
from os.path import join, isfile
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from ..utils.common import make_sub_dir

COLORS = []
for name, hex in matplotlib.colors.cnames.items():
    COLORS.append(name)
np.random.shuffle(COLORS)


class VesselTree(object):

    def __init__(self, img, skel, od_center, save_dir, name):

        self.img = img
        self.skel = skel
        self.od_center = od_center
        self.save_dir = save_dir
        self.name = name
        self.pkl_out = join(self.save_dir, self.name + '.pkl')

        self.font = {'color': 'white', 'size': 6}

        self.branch_dir = make_sub_dir(self.save_dir, 'branches')
        self.points_dir = make_sub_dir(self.save_dir, 'points')
        self.ratio_dir = make_sub_dir(self.save_dir, 'ratios')
        self.local_tort_dir = make_sub_dir(self.save_dir, 'local_tortuosity')

        self.branches = []
        self.visited = []
        self._get_end_points()

    def _get_end_points(self):

        self.end_points = []
        self.bifurcations = []

        conn_img = np.zeros_like(self.skel)

        for x, y in np.argwhere(self.skel):

            nb = np.sum(self.skel[x - 1:x + 2, y - 1:y + 2])
            conn_img[x, y] = nb

            if nb == 2:
                self.end_points.append((x, y))

            elif nb > 4:
                self.bifurcations.append((x, y))

        # Sort by distance from the optic disk
        dist = [vec_length(np.asarray(pt) - np.asarray(self.od_center)) for pt in self.end_points]
        self.end_points = [self.end_points[i] for i in np.argsort(dist)]

        # ep_arr = np.asarray(self.end_points)
        #
        # plt.imshow(self.img, cmap='gray')
        # plt.scatter(self.od_center[1], self.od_center[0], s=50)
        # plt.scatter(ep_arr[:, 1], ep_arr[:, 0], c=dist, cmap='hot')
        #
        # for i, pt in enumerate(self.end_points):
        #     plt.text(pt[1], pt[0], '{}'.format(i), fontdict={'color': 'white'})
        #
        # plt.show()

    def run(self):

        for x, y in self.end_points: # + self.bifurcations:

            traced = self.trace_branch(x, y, [])
            self.branches.extend(traced)

        # Save plot of branches
        self.plot_branches(out_name=join(self.branch_dir, self.name))
        return self.branches, self.get_features()

    def get_features(self):

        # Chord/arc length ratio
        # Number of crossings
        # Distance from chord
        # Vessel thickness

        lt_features = []
        # for i, c in enumerate(range(4, 24, 4)):

        local_tort = self.local_tortuosity(chord_length=10)
        lt_features = fit_gmm(local_tort, 'LT')

        # lr_features = self.length_ratios()
        # lr_features = self.fit_gmm(lr, 'CR')

        features = {'no_branches': len(self.branches)}
        features.update(lt_features)

        return features

    def trace_branch(self, x, y, parents):

        next_point = (x, y)  # initialise branch with start point
        if next_point in self.visited:
            return parents

        current_branch = [next_point]

        while next_point:

            px, py = next_point
            next_point = None

            coordinates = [(x, y) for x in range(px-1, px+2) for y in range(py-1, py+2)]

            for i, (nx, ny) in enumerate(coordinates):

                if (nx == px and ny == py) or self.skel[nx, ny] == 0\
                        or(nx, ny) in current_branch or (nx, ny) in self.visited:
                    continue

                # Add to visited list and current branch
                next_point = (nx, ny)
                current_branch.append(next_point)
                self.visited.append(next_point)

                if next_point in self.bifurcations:
                    parents.append(current_branch)
                    return self.trace_branch(nx, ny, parents=parents)
                elif next_point in self.end_points:
                    parents.append(current_branch)
                    return parents
                else:
                    break

        parents.append(current_branch)
        return parents

    def plot_branches(self, colors=None, out_name=None):

        fig, ax = plt.subplots()
        ax.imshow(self.img, cmap='gray', interpolation='none')

        if colors is None:
            base_colors = COLORS
            cycler = cycle(base_colors)
            colors = [next(cycler) for _ in range(0, len(self.branches))]

        for i, (b, col) in enumerate(zip(self.branches, colors)):

            b_arr = np.asarray(b)
            # ends = np.asarray(b)[[-1, 0]]
            # ax.text(ends[0, 1] - 5, ends[0, 0] - 5, 'E{}'.format(i), fontdict=self.font)
            # ax.text(ends[1, 1] - 5, ends[1, 0] - 5, 'S{}'.format(i), fontdict=self.font)
            ax.plot(b_arr[:, 1], b_arr[:, 0], c=col, lw=1.)

        bifurc_arr = np.asarray(self.bifurcations)
        end_arr = np.asarray(self.end_points)
        plt.scatter(end_arr[:, 1], end_arr[:, 0], s=10)

        try:
            plt.scatter(bifurc_arr[:, 1], bifurc_arr[:, 0], s=10)
        except IndexError:
            pass

        save_fig(ax, out_name)
        return fig, ax

    def chord_pairs(self, branch, chord_length):

        branch_points = np.asarray(branch)

        chord_range = list(range(chord_length, branch_points.shape[0] - chord_length, chord_length))
        if len(chord_range) == 0:
            yield None, None

        chord_1 = branch_points[[0, 0 + chord_length]]

        for i, pos in enumerate(chord_range):
            chord_2 = branch_points[[pos, pos + chord_length]]

            yield chord_1, chord_2
            chord_1 = chord_2

    def local_tortuosity(self, chord_length=10):

        fig, ax = plt.subplots()
        ax.imshow(self.img, cmap='gray')

        branch_tortuosities = []
        for branch in self.branches:

            vessel_angles = []
            for chord_1, chord_2 in self.chord_pairs(branch, chord_length):

                if chord_1 is None:
                    break

                plt.plot(chord_1[:, 1], chord_1[:, 0])
                plt.plot(chord_2[:, 1], chord_2[:, 0])

                a = angle_between(chord_1[1] - chord_1[0], chord_2[1] - chord_2[0])
                vessel_angles.append(a)
                plt.text(chord_2[0, 1], chord_2[0, 0], '{:.2f}'.format(a), fontdict=self.font)

            if len(vessel_angles) > 0:
                branch_tortuosities.append(sum(vessel_angles) / float(len(vessel_angles)))

        save_fig(ax, join(self.local_tort_dir, self.name))
        return branch_tortuosities

    def length_ratios(self):

        fig, ax = plt.subplots()
        ax.imshow(self.img, cmap='gray')

        ratios = []
        for i, b in enumerate(self.branches):

            b_arr = np.asarray(b)
            ax.plot(b_arr[:, 1], b_arr[:, 0], lw=1.)

            chord = b_arr[-1] - b_arr[0]  # first and last point in the branch
            ax.plot(b_arr[[0, -1], 1], b_arr[[0, -1], 0], 'r:', lw=2.)

            chord_len = vec_length(chord)

            if chord_len > 0:
                x = len(b) / chord_len
            else:
                x = 1.0

            ax.text(b_arr[0, 1], b_arr[0, 0], '{:.2f}'.format(x), fontdict=self.font)
            ratios.append(x)

        save_fig(ax, join(self.ratio_dir, self.name))

        return ratios


def fit_gmm(data, prefix):

    # Fit GMM to histogram
    X, _ = np.histogram(data, normed=True, bins=10)
    X = np.expand_dims(X, 1)

    gmm = GaussianMixture(n_components=2).fit(X)

    # Get means / variances in order of ascending mean
    mu = np.array([gmm.means_[0][0], gmm.means_[1][0]])
    var = np.array([gmm.covariances_[0][0], gmm.covariances_[1][0]])
    order = np.argsort(mu)

    # Means/variances of each component
    mu_0, mu_1 = mu[order]
    var_0, var_1 = var[order]

    return {'{}_mu0'.format(prefix): mu_0,
            '{}_mu1'.format(prefix): mu_1,
            '{}_var0'.format(prefix): var_0[0],
            '{}_var1'.format(prefix): var_1[0]}


def save_fig(ax, out_name):

    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if out_name:
        plt.savefig(out_name + '.svg', bbox_inches='tight', pad_inches=0)
        plt.close()