import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def create_subfigure(x_start, seq1, ax1, length, y, h, color):
    x = x_start
    label = seq1[0]
    seq1.append('none')
    start = 0
    for i in range(len(seq1)):
        if seq1[i] != label:
            w = float(i - start) / (len(seq1) - 1) * length
            start = i
            ax1.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    w,  # width
                    h,  # height
                    facecolor=color[label]#,
                    #edgecolor='black'
                )
            )
            x = x + w
            label = seq1[i]


def save_qualitative_result(scores, f_name, save_path, color, images = None):
    last_frame = len(scores[0])
    for i in range(1, len(scores)):
        last_frame = min(last_frame, len(scores))
    for score in scores:
        score = score[:last_frame]

    x_start = 3.5
    ys = [0.65, 0.58, 0.51, 0.44]
    #y_gt = .65
    h = 0.05
    length = 10

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 14])

    ax1.text(0, .670, 'I3D Features + TCN:', fontsize=6.5)
    ax1.text(0, .600, 'Output Retraining + TCN:', fontsize=6.5)
    ax1.text(0, .530, 'BoW-Network + TCN:', fontsize=6.5)
    ax1.text(0, .460, 'Ground Truth:', fontsize=6.5)

    create_subfigure(x_start, scores[0], ax1, length, ys[0], h, color)
    create_subfigure(x_start, scores[1], ax1, length, ys[1], h, color)
    create_subfigure(x_start, scores[2], ax1, length, ys[2], h, color)
    create_subfigure(x_start, scores[3], ax1, length, ys[3], h, color)


    ax1.axis('off')
    fig1.savefig(save_path+'/'+str(f_name.split('/')[-1])+'.pdf', dpi=150, bbox_inches='tight')

    

    plt.close()

    return