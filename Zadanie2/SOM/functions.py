import numpy as np
from scipy.spatial import distance


def Euklides_dist(x1, x2):
    """
    Euclidean distance:
    Square root of sum of squared substraction of every dimension
    """
    return distance.euclidean(x1, x2)


def RNF(R, distance):
    """
    rectangular neighborhood function
    R -  neighborhood radius
    distance - distance between dwo neurons in
    network topology

    Where R=0, this it winner takes all function
    """
    if distance <= R:
        return 1.0
    else:
        return 0.0


def GNF(R, distance):
    """
    Gaussian neighborhood function
    Args:
        R -  neighborhood radius
        distance - distance between dwo neurons in
        network topology
    """
    return np.exp(-(distance**2) / (2 * R**2))


def print_net(net):
    """
    Print network's neurons in console
    """
    for pos_net, pos_space in net.items():
        print("pos_net", pos_net)
        print("pos_space", pos_space)
    print()


def quantization_error(neurons, points, dist_func=Euklides_dist):
    """
    Args:
        Neurons: list of neuron posion in space
        points: list of point posion in space
        dist_func: callable object that compute distance in space
    Return:
        Mean of distances from every neuron to every point
    """
    err = 0
    for neuron in neurons:
        for point in points:
            err += dist_func(neuron, point)
    return err / len(neurons)


def quantization_error2(neurons, points, dist_func=Euklides_dist):
    """
    Args:
        Neurons: list of neuron posion in space
        points: list of point posion in space
        dist_func: callable object that compute distance in space
    Return:
        Mean of distances from every neuron to corresponding closest point
    """
    err = 0
    for neuron in neurons:
        # the closest point
        tcp = min(points, key=lambda p: dist_func(neuron, p))
        err += dist_func(neuron, tcp)
    return err / len(neurons)


def quantization_error3(neurons, points, dist_func=Euklides_dist):
    """
    Args:
        Neurons: list of neuron posion in space
        points: list of point posion in space
        dist_func: callable object that compute distance in space
    Return:
        Mean of distances from every neuron to corresponding closest point
    """
    err = 0
    for point in points:
        # find the closest neuron
        tcn = min(neurons, key=lambda n: dist_func(point, n))
        err += dist_func(point, tcn)
    return err / len(neurons)


def random_point(n):
    """
    Generete random point in n-dimensional space
    Returns:
            n - element tuple of random 32-bit floats
    """
    # bounds are temporarily hardcoded to -10,10
    return tuple(np.random.uniform(low=-10, high=10, size=(n)))


def voronoi_finite_polygons_2d(vor, radius=None):
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
