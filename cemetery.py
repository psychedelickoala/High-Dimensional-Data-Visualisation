    def plot_on_plane(self, plane: tuple[np.ndarray] | str = "optimised 0", m_dists: list[float] = [1, 2, 3], show_axes: bool = True,\
        points_only: bool = False, opt_step: float = 0.005, opt_tol: float = 0.0001, verbose: bool = False) -> list[np.ndarray] | None:
        """
        Plots data onto a plane.

        :optional param plane: tuple of dim x 1 vectors spanning a plane, else "optimised [cutoff]" or "random"; plane to project onto. Default "optimised 0"
        :optional param m_dists: list of floats; Mahalanobis distances to draw ellipse. Default [1, 2, 3]
        :optional param show_axes: boolean; whether or not to plot axes in grey. Default True.
        :optional param points_only: boolean; whether or not to return an array of ellipse points rather than plotting. Default False.
        :optional param opt_step: small float; sensitivity of steps in optimisation method. Smaller values cost computation time. Default 0.005
        :optional param opt_tol: small float; tolerance of optimisation method. Default 0.0001
        :optional param verbose: whether or not to print progress. Default False.

        :return: list of 2 x ellipse_res arrays of points of an ellipse if points_only is True, else None.
        """
        # set colours
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#fecb3e", "#fc8370", "#c2549d", "#7e549e"])

        # set plane
        cutoff = None
        if type(plane) is str and "optimise" in plane:
            cutoff = float(plane.split()[-1])
            u, v = self.optimise_plane(cutoff=cutoff, step=opt_step, tol = opt_tol, verbose=verbose)
            P = np.vstack([u, v])
        elif plane == "random":
            u = np.random.rand(self.__dim) - 0.5
            v = np.random.rand(self.__dim) - 0.5
            u, v = self.orthonormalise(u, v)
            P = np.vstack([u, v])
        else:
            u, v = self.orthonormalise(plane[0], plane[1])
            P = np.vstack([u, v])

        # get ellipses
        M = P @ self.__covariance @ P.T
        T = np.linalg.cholesky(M)
        proj_mean = P @ self.__cov_mean
        ellipses = [(dist * T @ self.__circle) + proj_mean for dist in m_dists]

        # return points
        if points_only:
            return ellipses

        # plot axes
        if show_axes:
            for i in range(self.__dim):
                plt.plot([0, P.T[i][0]], [0, P.T[i][1]], c = "grey", linewidth = 1)

        # plot points
        proj_data = P @ self.__data
        if cutoff is not None:
            ind = np.searchsorted(self.__sds, cutoff)
            plt.scatter(proj_data[0, :ind], proj_data[1, :ind], c = "#808080", marker = ".")
            plt.scatter(proj_data[0, ind:], proj_data[1, ind:], c = "#000000", marker = ".")
        else:
            plt.scatter(proj_data[0], proj_data[1], c = "#000000", marker = ".")

        # plot ellipses
        for i in range(len(ellipses)):
            plt.plot(ellipses[i][0], ellipses[i][1])
        
        plt.show()
        return P[0], P[1]
    
    def plot_comparison(self, optimise_cutoffs: list[float] = [0, 3], across: int = 2, down: int = 2, m_dists:list[float] = [1, 2, 3],\
        show_axes: bool = True, opt_step: float = 0.005, opt_tol: float = 0.0001, verbose: bool = False) -> None:
        """
        Plots an across x down collection of viewpoints, some optimised and some random.

        :optional param optimise_cutoffs: list of floats; cutoffs for different optimised viewpoints
        :optional param across: integer number of plots to put across
        :optional param down: integer number of plots to put down
        :optional param m_dists: list of floats; Mahalanobis distances to draw ellipses. Default [1, 2, 3]
        :optional param show_axes: boolean; whether or not to plot axes in grey. Default True.
        :optional param opt_step: small float; sensitivity of steps in optimisation method. Smaller values cost computation time. Default 0.005
        :optional param opt_tol: small float; tolerance of optimisation method. Default 0.0001
        :optional param verbose: whether or not to print progress. Default False.
        """

        # for testing !
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#fecb3e", "#fc8370", "#c2549d", "#7e549e"])
        planes = []

        factor = 1.01
        mod = np.log(factor)

        # get optimised plane
        for i in range(down):
            row = []
            for _ in range(across):
                u = np.random.rand(self.__dim) - 0.5
                v = np.random.rand(self.__dim) - 0.5
                u, v = self.orthonormalise(u, v)
                print(f"random dist = {self.total_m_dist(u, v, self.__data - self.__mean, mod)}")
                P = np.vstack([u, v])
                row.append(P)
            planes.append(row)
        
        from_plane = None
        for i, cutoff in enumerate(optimise_cutoffs):
            u, v = self.optimise_plane(cutoff=cutoff, from_plane=from_plane, step=opt_step, tol=opt_tol, factor=factor, verbose=verbose)
            P = np.vstack([u, v])
            planes[i//across][i%across] = P
            from_plane = (u, v)
        
        fig, axs = plt.subplots(down, across)

        for j in range(down):
            for i in range(across):
                P = planes[j][i]
                this_cutoff = 0
                if j*across + i < len(optimise_cutoffs):
                    if verbose:
                        print(f"plotting optimised {optimise_cutoffs[j*across + i]}")
                    axs[j, i].set_title(f"optimised {optimise_cutoffs[j*across + i]}")
                    this_cutoff = optimise_cutoffs[j*across + i]
                else:
                    if verbose:
                        print("plotting random")
                    axs[j, i].set_title(f"random")
                #M = np.linalg.inv(P @ self.__ellipsoid @ P.T)#
                M= P @ self.__covariance @ P.T
                T = np.linalg.cholesky(M)
                proj_mean = P @ self.__mean
                ellipses = [(dist * T @ self.__circle) + proj_mean for dist in m_dists]

                # plot axes
                if show_axes:
                    for k in range(self.__dim):
                        axs[j, i].plot([0, P.T[k][0]], [0, P.T[k][1]], c = "#505050", linewidth = 1)

                # plot points
                proj_data = P @ self.__data
                ind = np.searchsorted(self.__sds, this_cutoff)
                axs[j, i].scatter(proj_data[0][:ind], proj_data[1][:ind], c = "#aaaaaa", marker = ".")
                axs[j, i].scatter(proj_data[0][ind:], proj_data[1][ind:], c = "#000000", marker = ".")

                # plot ellipses
                for k in range(len(ellipses)):
                    axs[j, i].plot(ellipses[k][0], ellipses[k][1])

        plt.show()


# Lasso
class SelectFromCollection:

    def __init__(self, ax, collection, graph_obj):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.graph = graph_obj

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        self.fc = self.graph.point_colours


        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:] = self.graph.Palette.redundant_points_colour
        self.fc[self.ind] = self.graph.Palette.points_colour
        self.graph.point_colours = self.fc
        self.graph.update()
        self.graph.ax.set_title("Press enter to accept selection")
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        #cutoff_ind = self.graph.CALC.partition_data(InteractiveGraph.cutoff_slider.val)
        #self.fc[:cutoff_ind] = self.graph.Palette.redundant_points_colour
        #self.fc[cutoff_ind:] = self.graph.Palette.points_colour
        #self.collection.set_facecolors(self.fc)
        #self.graph.change_cutoff()
        self.canvas.draw_idle()


        readfile = input("Enter path to data csv file: ")
        if readfile == "1":
            readfile = "samples/p5pexample/np.csv"
            cov = np.loadtxt('samples/p5pexample/cov_mat.csv', dtype=float, delimiter=",")
            mean = np.loadtxt('samples/p5pexample/centre_of_ellipses.csv', dtype=float, delimiter=",")
        else:
            manual_cov = input("Would you like to specify a covariance matrix? Y/N: ")
            cov = mean = None
            if manual_cov == 'Y':
                cov_file = input("Enter path to covariance matrix file: ")
                cov = np.loadtxt(cov_file, dtype=float, delimiter=",")
                manual_mean = input("Would you like to specify an ellipsoid mean? Y/N: ")
                if manual_mean == "Y":
                    mean_file = input("Enter path to ellipsoid mean file: ")
                    mean = np.loadtxt(mean_file, dtype=float, delimiter=",")