import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

TNSE_AX_Q = 0.999

MARKER_SIZE = 10

SUBPLOT_SIZE = 4

# '#f2f2f2' #(0.98, 0.98, 0.98) #(0.8, 0.8, 0.8) #(0.85, 0.85, 0.85
BACKGROUND_SAMPLE_COLOR = [0.75, 0.75, 0.75]
EDGE_COLOR = None  # [0.3, 0.3, 0.3] #'#4d4d4d'
EDGE_WIDTH = 0  # 0.25
ALPHA = 0.9

BLUE_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'blue_yellow', ['#162d50', '#ffdd55'])
BLUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'blue', ['#162d50', '#afc6e9'])
BLUE_GREEN_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bgy', ['#162d50', '#214478', '#217844', '#ffcc00', '#ffdd55'])

# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#002255', '#2ca05a', '#ffd42a'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#002255', '#003380', '#2ca05a', '#ffd42a', '#ffdd55'])

# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#003366', '#339966', '#ffff66', '#ffff00')
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#001a33', '#003366', '#339933', '#ffff66', '#ffff00'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#00264d', '#003366', '#339933', '#e6e600', '#ffff33'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#003366', '#40bf80', '#ffff33'])

BGY_ORIG_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bgy', ['#002255', '#003380', '#2ca05a', '#ffd42a', '#ffdd55'])

BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bgy', ['#003366', '#004d99', '#40bf80', '#ffe066', '#ffd633'])

GRAY_PURPLE_YELLOW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'grey_purple_yellow', ['#e6e6e6', '#3333ff', '#ff33ff', '#ffe066'])

GYBLGRYL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'grey_blue_green_yellow', ['#e6e6e6', '#0055d4', '#00aa44', '#ffe066'])

OR_RED_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'or_red', matplotlib.cm.OrRd(range(4, 256)))

BU_PU_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bu_pu', matplotlib.cm.BuPu(range(4, 256)))


# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#0066ff', '#37c871', '#ffd42a'])
# BGY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('bgy', ['#003380', '#5fd38d', '#ffd42a'])

EXP_NORM = matplotlib.colors.Normalize(-1, 3, clip=True)

LEGEND_PARAMS = {'show': True, 'cols': 4, 'markerscale': 2}


CLUSTER_101_COLOR = (0.3, 0.3, 0.3)



PATIENT_082917_COLOR = 'mediumorchid'
PATIENT_082917_EDGE_COLOR = 'purple'

PATIENT_082217_COLOR = 'gold'
PATIENT_082217_EDGE_COLOR = 'goldenrod'

PATIENT_011018_COLOR = 'mediumturquoise'
PATIENT_011018_EDGE_COLOR = 'darkcyan'

PATIENT_013118_COLOR = 'salmon'
PATIENT_013118_EDGE_COLOR = 'darkred'

EDGE_COLOR = 'dimgray'

C3_COLORS = ['tomato', 'mediumseagreen', 'royalblue']
EDGE_COLORS = ['darkred', 'darkgreen', 'darkblue']



PCA_RANDOM_STATE = 0