from geom import *
from os.path import join, isfile
from sklearn.mixture import GaussianMixture
import numpy as np
import pickle


class VesselTree(object):

    def __init__(self, img, skel, save_dir, name):

        self.img = img
        self.skel = skel
        self.save_dir = save_dir
        self.name = name
        self.pkl_out = join(self.save_dir, self.name + '.pkl')

        self.branches = []
        self.visited = []
        self._get_end_points()

    def _get_end_points(self):

        self.end_points = []
        self.bifurcations = []

        for x, y in np.argwhere(self.skel):

            nb = np.sum(self.skel[x - 1:x + 2, y - 1:y + 2])

            if nb == 2:
                self.end_points.append((x, y))

            elif nb > 4:
                self.bifurcations.append((x, y))

    def run(self):

        if isfile(self.pkl_out):
            return pickle.load(open(self.pkl_out, 'rb'))

        for x, y in self.end_points + self.bifurcations:

            traced = self.trace_branch(x, y, [])
            self.branches.extend(traced)

        self.plot_branches()
        return self.get_features()

    def get_features(self):

        # Chord length ratio
        # Number of crossings
        # Distance from chord
        # Vessel thickness

        length_ratios = []
        for b in self.branches:

            start_end = np.asarray(b[-1]) - np.asarray(b[0])
            v_len = vec_length(start_end)

            if v_len > 0:
                x = vec_length(start_end) / len(b)
                length_ratios.append(x)

        mu_0, mu_1, var_0, var_1 = self.fit_gmm(length_ratios)
        features = {'total': len(self.branches), 'mu_0': mu_0, 'mu_1': mu_1, 'var_0': var_0, 'var_1': var_1}

        with open(self.pkl_out, 'wb') as pkl:
            pickle.dump(features, pkl)

        return features

    def fit_gmm(self, data):

        # Fit GMM to histogram
        X, _ = np.histogram(data, normed=True, bins=10)
        X = np.expand_dims(X, 1)

        gmm = GaussianMixture(n_components=2).fit(X)

        # Get means / variances in order of ascending mean
        mu = np.array([gmm.means_[0][0], gmm.means_[1][0]])
        var = np.array([gmm.covariances_[0][0], gmm.covariances_[1][0]])
        order = np.argsort(mu)

        mu_0, mu_1 = mu[order]
        var_0, var_1 = var[order]

        return mu_0, mu_1, var_0, var_1

    def trace_branch(self, x, y, parents):

        next_point = (x, y)  # initialise branch with end point
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

    def plot_branches(self):

        fig, ax = plt.subplots()
        ax.imshow(self.img, cmap='gray')

        for b in self.branches:
            b_arr = np.asarray(b)
            ax.plot(b_arr[:, 1], b_arr[:, 0])

        # Plot end points/bifurcations
        # bifurc_arr = np.asarray(self.bifurcations)
        # end_arr = np.asarray(self.end_points)
        # plt.scatter(end_arr[:, 1], end_arr[:, 0], s=3)
        # plt.scatter(bifurc_arr[:, 1], bifurc_arr[:, 0], s=3)

        ax.axis('off')
        fig.savefig(join(self.save_dir, self.name + '.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

