"""
Created by: Paul Aljabar
Date: 08/06/2023
"""

import random
import  matplotlib.colors as mcolors


colors_A = list(mcolors.TABLEAU_COLORS.keys())
colors_B = list(mcolors.CSS4_COLORS.keys())
ix = list(range(len(colors_B)))
random.seed(213)
random.shuffle(ix)
colors_C = [colors_B[k] for k in ix]
COLORS = colors_A + colors_C
N_COLORS = len(COLORS)

def get_color_list(N):
    if N < N_COLORS:
        return COLORS[:N]
    ret = []
    while N > N_COLORS:
        ret += COLORS
        N -= N_COLORS
    ret += COLORS[:N]
    return ret

def get_Nth_colour(N):
    k = (N-1) % N_COLORS
    return COLORS[k]

