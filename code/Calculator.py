import numpy as np


def get_degree_arc(alpha):
    return alpha * 2 * np.pi / 360


def get_degree_angle(alpha):
    return alpha * 360 / (2 * np.pi)


def r2xy(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def xy2r(x, y):
    return np.sqrt(x**2, y**2), np.arctan(y / x)


def is_equal(a: np.float64, b: np.float64) -> np.bool:
    return np.abs(a - b) < 1e-7


def make_line(p1, p2):
    k = np.float64(p2[1] - p1[1]) / np.float64(p2[0] - p1[0])
    vk = -1 / k
    b = p1[1] - p1[0] * k
    vb = (p2[1] + p1[1]) / 2 - vk * (p2[0] + p1[0]) / 2
    alpha = np.arctan(k)
    return k, b, vk, vb, alpha, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def get_line_length(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def get_side_of_line(l, p):
    k, b, vk, vb, alpha, x, y = l
    if k != np.inf and k != -np.inf:
        return p[1] > (k * p[0] + b)
    else:
        return p[0] > x


def is_same_sade_of_circle(l, p, center1, center2, alpha):
    if get_side_of_line(l, p) == get_side_of_line(l, center1):
        if alpha <= np.pi / 2:
            return center1
        else:
            return center2
    elif get_side_of_line(l, p) == get_side_of_line(l, center2):
        if alpha <= np.pi / 2:
            return center2
        else:
            return center1
    else:
        return None


def routate(p: np.ndarray, alpha):
    return np.dot(
        p, np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    )


def get_angle(p1, p2, p):
    a = get_line_length(p1, p)
    b = get_line_length(p2, p)
    c = get_line_length(p1, p2)
    return np.arccos((a**2 + b**2 - c**2) / (2 * a * b))


def location(ps, p):
    alpha_1 = get_angle(ps[0], ps[1], p)
    alpha_2 = get_angle(ps[0], ps[2], p)
    alpha_3 = get_angle(ps[2], ps[1], p)
    return alpha_1, alpha_2, alpha_3


def get_center(p1, p2, theta, pr):
    upset, p1, p2 = (False, p1, p2) if p1[0] <= p2[0] else (True, p2, p1)
    k, b, vk, vb, alpha, x, y = make_line(p1, p2)

    l = get_line_length(p1, p2)
    radius = l / (2 * np.sin(theta))

    center_origin_p = np.array([0, radius * np.cos(theta)])
    center_origin_n = np.array([0, -radius * np.cos(theta)])

    p2_origin = np.array([l / 2, 0])

    p_diff = p2 - routate(p2_origin, alpha)

    center_p = routate(center_origin_p, alpha) + p_diff
    center_n = routate(center_origin_n, alpha) + p_diff

    return (
        is_same_sade_of_circle(
            (k, b, vk, vb, alpha, x, y), pr, center_n, center_p, theta
        ),
        radius,
    )
