import matplotlib
import matplotlib.pyplot as plt

CLUSTER_101_COLOR = (0.3, 0.3, 0.3)

BLUE_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_yellow", ["#162d50", "#ffdd55"]
)
BLUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue", ["#162d50", "#afc6e9"]
)
BLUE_GREEN_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bgy", ["#162d50", "#214478", "#217844", "#ffcc00", "#ffdd55"]
)

# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#002255', '#2ca05a', '#ffd42a'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#002255', '#003380', '#2ca05a', '#ffd42a', '#ffdd55'])

# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#003366', '#339966', '#ffff66', '#ffff00')
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#001a33', '#003366', '#339933', '#ffff66', '#ffff00'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#00264d', '#003366', '#339933', '#e6e600', '#ffff33'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#003366', '#40bf80', '#ffff33'])

BGY_ORIG_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bgy", ["#002255", "#003380", "#2ca05a", "#ffd42a", "#ffdd55"]
)

BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bgy", ["#003366", "#004d99", "#40bf80", "#ffe066", "#ffd633"]
)

BGY2_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bgy", ["#003366", "#004d99", "#40bf80", "#ffe066", "#ffcc00"]
)

GRAY_PURPLE_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey_purple_yellow", ["#e6e6e6", "#3333ff", "#ff33ff", "#ffe066"]
)

GRAY_PURPLE_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey_purple_yellow", ["#e6e6e6", "#3333ff", "#ff33ff", "#ffe066"]
)

GYBLGRYL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey_blue_green_yellow", ["#e6e6e6", "#0055d4", "#00aa44", "#ffe066"]
)

OR_RED_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "or_red", matplotlib.cm.OrRd(range(4, 256))
)

BU_PU_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bu_pu", matplotlib.cm.BuPu(range(4, 256))
)

VIRIDIS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "viridis", matplotlib.cm.viridis(range(0, 240))
)

GRAY_REDS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey_red",
    [
        "#e6e6e6",
        "#ff9999",
        "#ff8080",
        "#ff6666",
        "#ff4d4d",
        "#ff3333",
        "#ff1a1a",
        "#ff0000",
        "#e60000",
        "#cc0000",
    ],
)


def get_colors():
    """
    Make a list of usable colors and include 101 as an entry for
    questionable clusters
    Unassigned colors between 0 and 101 are black
    """
    ret = [(0, 0, 0)] * 102

    c = 0

    # l = list(plt.cm.tab20c.colors)
    # for i, color in enumerate(l):
    #     ret[c] = color
    #     c += 1

    l = list(plt.cm.tab10.colors)
    for i, color in enumerate(l):
        if i == 7 or i == 8:
            continue
        ret[c] = color
        c += 1

    l = list(plt.cm.Dark2.colors)
    for color in l[0:-1]:
        ret[c] = color
        c += 1

    l = list(plt.cm.Set2.colors)
    for color in l[0:-1]:
        ret[c] = color
        c += 1

    l = list(plt.cm.Pastel1.colors)
    for color in l[0:-1]:
        ret[c] = color
        c += 1

    # for i in range(0, 20, 2):
    #     # skip gray
    #     if i == 14:
    #         continue

    #     ret[c] = l[i]
    #     c += 1

    # for i in range(0, 20, 2):
    #     if i == 14:
    #         continue

    #     ret[c] = l[i + 1]
    #     c += 1

    # ret = list(plt.cm.tab10.colors)
    # ret.extend(list(plt.cm.Set3.colors))

    # for color in list(plt.cm.Set3.colors):
    #     ret[c] = color
    #     c += 1

    # for color in list(plt.cm.Pastel1.colors):
    #     ret[c] = color
    #     c += 1

    ret[101] = CLUSTER_101_COLOR

    # ret.extend(list(plt.cm.Dark2.colors))
    # ret.extend(list(plt.cm.Set2.colors))

    return ret  # np.array(ret)


def colormap(n=-1):
    c = get_colors()

    if n > 0:
        c = c[0:n]

    return mcolors.ListedColormap(c, name="cluster")
