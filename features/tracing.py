from geom import *
from os.path import join


class VesselTree(object):

    def __init__(self, img, skel, save_dir, name):

        self.img = img
        self.skel = skel
        self.save_dir = save_dir
        self.name = name

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

        for x, y in self.end_points + self.bifurcations:

            traced = self.trace_branch(x, y, [])
            self.branches.extend(traced)

        self.plot_branches()

    def get_features(self):

        for b in self.branches:
            print b

    def trace_branch(self, x, y, parents):

        next_point = (x, y)
        current_branch = [next_point]

        while next_point:

            if next_point in self.visited:
                break

            px, py = next_point
            self.visited.append(next_point)
            next_point = None

            coordinates = [(x, y) for x in range(px-1, px+2) for y in range(py-1, py+2)]

            for i, (nx, ny) in enumerate(coordinates):

                if self.skel[nx, ny] == 0 or (nx, ny) in current_branch or (nx == px and ny == py):
                    continue

                # Add to visited list and current branch
                next_point = (nx, ny)
                current_branch.append(next_point)

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
        fig.savefig(join(self.save_dir, self.name), bbox_inches='tight', pad_inches=0)
        plt.close()
