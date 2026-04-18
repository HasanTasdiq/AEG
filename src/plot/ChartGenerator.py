from math import log10
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker


# ── Algorithm display names ───────────────────────────────────────────────────
# Order must match the order algorithms are written to the data files by Run.py.
ALGO_NAMES = ["AEG-LS", "ILP", "Random", "SP", "AEG-EC", "AEG-PES"]


class ChartGenerator:
    """Generate a single paper-quality figure from one .txt data file."""

    def __init__(self, dataName, Ylabel, Xlabel):
        filename = './data/' + dataName

        # ── Axis label formatting ─────────────────────────────────────────────
        if Ylabel in ('successfulRequest', '#successRequest'):
            Ylabel = '  Successful Request (%)  '
        if Ylabel == 'algorithmRuntime':
            Ylabel = 'Algorithm Runtime (s)'

        if Xlabel == '#RequestPerRound':
            Xlabel = '# Request Per Time Slot'
        if Xlabel == 'swapProbability':
            Xlabel = 'Swap Probability'
        if Xlabel == 'entanglementLifetime':
            Xlabel = 'Ent. Lifetime (Time slot)'
        if Xlabel == 'Timeslot':
            Xlabel = 'Time slot'
        if Xlabel == 'alpha':
            Xlabel = r'alpha ($\times 10^{-4}$)'

        if not os.path.exists(filename):
            print(f"[ChartGenerator] file doesn't exist: {filename}")
            return

        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"[ChartGenerator] generating {filename}")

        # ── Plot style ────────────────────────────────────────────────────────
        colors = [
            "#FF8800",   # AEG-LS  (index 0 — matches ALGO_NAMES[0])
            "#FF0000",   # ILP
            "#00AA00",   # Random
            "#0000FF",   # SP
            "#AA00AA",   # AEG-EC
            "#000000",   # AEG-PES
            "#900321",
            "#643321",
        ]
        markers = ['o', 's', 'v', 'x', 'd', '1', '<', '*']

        fontsize          = 30
        Xlabel_fontsize   = fontsize
        Ylabel_fontsize   = fontsize
        Xticks_fontsize   = 22
        Yticks_fontsize   = fontsize
        legSize           = 22

        andy_theme = {
            "xtick.labelsize":   20,
            "ytick.labelsize":   20,
            "axes.labelsize":    20,
            "axes.titlesize":    20,
            "font.family":       "Times New Roman",
            "mathtext.it":       "Times New Roman:italic",
            "mathtext.fontset":  "custom",
        }
        matplotlib.rcParams.update(andy_theme)

        fig, ax1 = plt.subplots(figsize=(7, 6), dpi=600)
        ax1.tick_params(direction="in")
        ax1.tick_params(bottom=True, top=True, left=True, right=True)
        ax1.tick_params(pad=20)

        # ── Parse data file ───────────────────────────────────────────────────
        # Format (written by Run.py):
        #   <x_value> <algo1_value> <algo2_value> ... \n
        x  = []
        _y = []
        numOfData = 0

        for line in lines:
            line = line.replace('\n', '')
            data = line.split(' ')
            numOfData += 1
            for i, token in enumerate(data):
                if i == 0:
                    x.append(token)
                else:
                    _y.append(token)

        numOfAlgo = len(_y) // numOfData if numOfData else 0
        if numOfAlgo == 0:
            print(f"[ChartGenerator] empty data in {filename}")
            plt.close()
            return

        y = [[] for _ in range(numOfAlgo)]
        for i in range(numOfData * numOfAlgo):
            y[i % numOfAlgo].append(_y[i])

        # ── Scale axes ────────────────────────────────────────────────────────
        Ypow = 0
        Xpow = 0
        for i in range(-10, -1):
            if float(x[numOfData - 1]) <= 10 ** i:
                Xpow = i - 2

        Ydiv = float(10 ** Ypow)
        Xdiv = float(10 ** Xpow)

        for i in range(numOfData):
            x[i] = float(x[i]) / Xdiv

        maxData = 0
        minData = math.inf
        for i in range(numOfAlgo):
            for j in range(numOfData):
                y[i][j] = float(y[i][j]) / Ydiv
                maxData  = max(maxData, y[i][j])
                minData  = min(minData, y[i][j])

        Yend = math.ceil(maxData)
        Ystart    = 0
        Yinterval = (Yend - Ystart) / 5

        if maxData > 1.1:
            Yinterval = int(math.ceil(Yinterval))
            Yend      = int(Yend)
        else:
            Yend      = 1
            Ystart    = 0
            Yinterval = 0.2

        # ── Plot lines ────────────────────────────────────────────────────────
        markers_on = list(range(len(x)))
        if len(markers_on) > 5:
            markers_on = _get_n_index(markers_on, 5)

        for i in range(numOfAlgo):
            ax1.plot(x, y[i],
                     color=colors[i % len(colors)],
                     markevery=markers_on,
                     lw=2.5,
                     linestyle="-",
                     marker=markers[i % len(markers)],
                     markersize=10,
                     markerfacecolor="none",
                     markeredgewidth=2.5)

        plt.xticks(fontsize=Xticks_fontsize)
        plt.yticks(fontsize=Yticks_fontsize)

        # ── Legend ────────────────────────────────────────────────────────────
        # Use as many names as there are actual algorithms in the data file.
        algo_names = ALGO_NAMES[:numOfAlgo]

        leg = plt.legend(
            algo_names,
            loc=10,
            bbox_to_anchor=(0.4, 1.25),
            prop={"size": legSize, "family": "Times New Roman"},
            frameon=False,
            labelspacing=0.2,
            handletextpad=0.2,
            handlelength=1,
            columnspacing=0.2,
            ncol=2,
            facecolor="None",
        )
        leg.get_frame().set_linewidth(0.0)

        # ── Axes labels and ticks ─────────────────────────────────────────────
        Ylabel += _gen_multi_name(Ypow)
        Xlabel += _gen_multi_name(Xpow)

        plt.subplots_adjust(top=0.75, left=0.3, right=0.95, bottom=0.25)
        plt.yticks(np.arange(Ystart, Yend + Yinterval, step=Yinterval),
                   fontsize=Yticks_fontsize)
        plt.xticks(x)
        plt.ylabel(Ylabel, fontsize=Ylabel_fontsize, labelpad=10)
        plt.xlabel(Xlabel, fontsize=Xlabel_fontsize, labelpad=10)
        plt.locator_params(axis='x', nbins=5)

        ax1.xaxis.set_label_coords(0.45, -0.27)

        # ── Save ─────────────────────────────────────────────────────────────
        os.makedirs('./pdf', exist_ok=True)   # create output dir if needed
        pdfName = dataName[:-4]               # strip .txt
        plt.savefig(f'./pdf/{pdfName}.jpg')
        plt.close()
        print(f"[ChartGenerator] saved ./pdf/{pdfName}.jpg")


# ── Module-level helpers ──────────────────────────────────────────────────────

def _gen_multi_name(multiple):
    """Return a LaTeX power-of-ten suffix string, e.g. '($10^{3}$)'."""
    if multiple == 0:
        return ''
    return f'($10^{{{multiple}}}$)'


def _get_n_index(sorted_list, n):
    """Return n evenly-spaced indices from sorted_list."""
    if len(sorted_list) < 2 or n < 2:
        return sorted_list[:n]

    common_diff = sorted_list[1] - sorted_list[0]
    step_size   = (sorted_list[-1] - sorted_list[0]) / (n - 1)

    result = []
    i = 0
    while len(result) < n:
        if i >= len(sorted_list):
            i = len(sorted_list) - 1
        result.append(i)
        i += int(round(step_size / common_diff))
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sweep X-axis labels that Run.py produces files for
    Xlabels = [
        "#RequestPerRound",
        "totalRequest",
        "#nodes",
        "r",
        "swapProbability",
        "alpha",
        "SocialNetworkDensity",
        "preSwapFraction",
        "entanglementLifetime",
        "requestTimeout",
        "preSwapCapacity",
    ]
    # Metric Y-axis labels (must match AlgorithmResult.Ylabels)
    Ylabels = [
        "algorithmRuntime",
        "waitingTime",
        "idleTime",
        "usedQubits",
        "temporaryRatio",
        "entanglementPerRound",
        "successfulRequest",
        "usedLinks",
    ]

    # Generate one chart per (Xlabel, Ylabel) combination
    for Xlabel in Xlabels:
        for Ylabel in Ylabels:
            ChartGenerator(Xlabel + '_' + Ylabel + '.txt', Ylabel, Xlabel)

    # Time-series charts
    ChartGenerator("Timeslot_#remainRequest.txt",  "#remainRequest",  "Timeslot")
    ChartGenerator("Timeslot_#successRequest.txt", "#successRequest", "Timeslot")
