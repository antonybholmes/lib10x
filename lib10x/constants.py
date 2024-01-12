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