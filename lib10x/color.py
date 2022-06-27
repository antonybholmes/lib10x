import matplotlib.pyplot as plt

CLUSTER_101_COLOR = (0.3, 0.3, 0.3)


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

  
    #ret = list(plt.cm.tab10.colors)
    #ret.extend(list(plt.cm.Set3.colors))
    

    # for color in list(plt.cm.Set3.colors):
    #     ret[c] = color
    #     c += 1    
    
    # for color in list(plt.cm.Pastel1.colors):
    #     ret[c] = color
    #     c += 1
    
    ret[101] = CLUSTER_101_COLOR
  
    #ret.extend(list(plt.cm.Dark2.colors))
    #ret.extend(list(plt.cm.Set2.colors))
  
    return ret #np.array(ret)


def colormap(n=-1):
    c = get_colors()
    
    if n > 0:
        c = c[0:n]
    
    return mcolors.ListedColormap(c, name='cluster')