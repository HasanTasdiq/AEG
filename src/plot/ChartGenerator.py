from cmath import log10
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.transforms
# import latex
import matplotlib
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

class ChartGenerator:
    # data檔名 Y軸名稱 X軸名稱 Y軸要除多少(10的多少次方) Y軸起始座標 Y軸終止座標 Y軸座標間的間隔
    def __init__(self, dataName, Ylabel, Xlabel):
        filename = './data/' + dataName
        if Ylabel == 'successfulRequest' or Ylabel == '#successRequest':
            Ylabel = '  Successful Request (%)  '        
        if Xlabel == '#RequestPerRound':
            Xlabel = '# Request Per Time Slot'
        if Xlabel == 'swapProbability':
            Xlabel = 'Swap Probability'
        if Xlabel == 'entanglementLifetime':
            Xlabel = 'Ent. Lifetime (Time slot)'
        if Xlabel == 'Timeslot':
            Xlabel = 'Time slot'
        if Ylabel == 'algorithmRuntime':
            Ylabel = 'Algorithm Runtime (s)'
        if not os.path.exists(filename):
            print("file doesn't exist")
            return
        
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print("start generate", filename)
        
        
        # Ydiv, Ystart, Yend, Yinterval
        Ypow = 0
        Xpow = 0

        # color = [
        #     "#000000",
        #     "#006400",
        #     "#FF1493",
        #     "#7FFF00",   
        #     "#900321",
        # ]
        color = [
            "#FF0000",
            "#00FF00",   
            "#0000FF",
            "#000000",
            "#900321",
            "#F00121",
            "#401321",
            "#643321",
        ]
        # matplotlib.rcParams['text.usetex'] = True

        fontsize = 30
        Xlabel_fontsize = fontsize
        Ylabel_fontsize = fontsize
        Xticks_fontsize = 22
        Yticks_fontsize = fontsize
        legSize = 22
            
        # matplotlib.rcParams['text.usetex'] = True
        # fig, ax = plt.subplots(figsize=(8, 6), dpi=600) 
        
        andy_theme = {
        # "axes.grid": True,
        # "grid.linestyle": "--",
        # "legend.framealpha": 1,
        # "legend.facecolor": "white",
        # "legend.shadow": True,
        # "legend.fontsize": 14,
        # "legend.title_fontsize": 16,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "font.family": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        # "mathtext.default": "regular",
        "mathtext.fontset": "custom"
        # "mathtext.fontset": "custom"
        # "figure.autolayout": True
        # "text.usetex": True,
        # "figure.dpi": 100,
        }
        
        matplotlib.rcParams.update(andy_theme)
        fig, ax1 = plt.subplots(figsize = (7, 6), dpi = 600)
        # ax1.spines['top'].set_linewidth(1.5)
        # ax1.spines['right'].set_linewidth(1.5)
        # ax1.spines['bottom'].set_linewidth(1.5)
        # ax1.spines['left'].set_linewidth(1.5)
        ax1.tick_params(direction = "in")
        ax1.tick_params(bottom = True, top = True, left = True, right = True)
        ax1.tick_params(pad = 20)

        ##data start##
        x = []
        _y = []
        numOfData = 0

        for line in lines:
            line = line.replace('\n','')
            data = line.split(' ')
            numOfline = len(data)
            numOfData += 1
            for i in range(numOfline):
                if i == 0:
                    x.append(data[i])
                else:
                    _y.append(data[i])        
        
        numOfAlgo = len(_y) // numOfData

        y = [[] for _ in range(numOfAlgo)]
        for i in range(numOfData * numOfAlgo):
            y[i % numOfAlgo].append(_y[i])

        # print(x)
        # print(y)

        maxData = 0
        minData = math.inf

        for i in range(-10, -1, 1):
            if float(x[numOfData - 1]) <= 10 ** i:
                Xpow = (i - 2)

        Ydiv = float(10 ** Ypow)
        Xdiv = float(10 ** Xpow)
        
        for i in range(numOfData):
            x[i] = float(x[i]) / Xdiv

        for i in range(numOfAlgo):
            for j in range(numOfData):
                y[i][j] = float(y[i][j]) / Ydiv
                maxData = max(maxData, y[i][j])
                minData = min(minData, y[i][j])

        Yend = math.ceil(maxData)
        Ystart = 0
        Yinterval = (Yend - Ystart) / 5

        if maxData > 1.1:
            Yinterval = int(math.ceil(Yinterval))
            Yend = int(Yend)
        else:
            Yend = 1
            Ystart = 0
            Yinterval = 0.2

        marker = ['o', 's', 'v', 'x', 'd' , '1' , '<' , '*']
        markers_on = [i for i in range(len(x))]
        if len(markers_on) > 5:
            # print(markers_on)
            markers_on = get_n_index(markers_on , 5)
        for i in range(numOfAlgo):
            ax1.plot(x, y[i], color = color[i], markevery=markers_on, lw = 2.5, linestyle = "-", marker = marker[i], markersize = 10, markerfacecolor = "none", markeredgewidth = 2.5)
        # plt.show()

        plt.xticks(fontsize = Xticks_fontsize)
        plt.yticks(fontsize = Yticks_fontsize)
        
        # AlgoName = ["SEER", "Greedy", "Q-CAST", "REPS"]
        # AlgoName = [ "Q-CAST", "Cache", "REPS"]
        # AlgoName = ["SEER","SEER-Cache", "SEER-Cache2", "Q-CAST", "Q-CAST-Cache", "REPS"]
        # AlgoName = ["REPS","REPS-CACHE", "REPS-RE-USE-SWAPP","REPS4-PRE-SWAP" ]
        # AlgoName = ["REPS","REPS-cache","REPS-preswap"]
        # AlgoName = ["SEER","SEER-cache", "SEER-1hop-pre-swap","SEER-multihop-pre-swap", "SEER-multihop-pre-swap-qrl","SEER-multihop-swap-dqrl"]
        # AlgoName = ["REPS","REPS-EC", "REPS-PEG-heuristic", "REPS-PEG-qrl"]
        # AlgoName = ["Original","Ent. Caching", "PES-heuristic", "PES-QRL", "PES-DeepQRL"]
        # AlgoName = ["Original","Ent. Caching", "PES-heuristic", "PES-QRL", "PES-DeepQRL" , '4' , '5' , '6' , '7' ,'8']
        AlgoName = ["Original","Ent. Caching", "PES-DeepQRL", "Original-SEG","Ent. Caching-SEG", "PES-DeepQRL-SEG"]
        # AlgoName = ["SEER","SEER-cache","SEER-preswap"]
        # AlgoName = ["SEER Ent. Caching" , "REPS Ent. Caching"]

        leg = plt.legend(
            AlgoName,
            loc = 10,
            bbox_to_anchor = (0.4, 1.25),
            prop = {"size": legSize, "family": "Times New Roman"},
            frameon = "False",
            labelspacing = 0.2,
            handletextpad = 0.2,
            handlelength = 1,
            columnspacing = 0.2,
            ncol = 2,
            facecolor = "None",
        )

        leg.get_frame().set_linewidth(0.0)
        Ylabel += self.genMultiName(Ypow)
        Xlabel += self.genMultiName(Xpow)
        plt.subplots_adjust(top = 0.75)
        plt.subplots_adjust(left = 0.3)
        plt.subplots_adjust(right = 0.95)
        plt.subplots_adjust(bottom = 0.25)

        plt.yticks(np.arange(Ystart, Yend + Yinterval, step = Yinterval), fontsize = Yticks_fontsize)
        plt.xticks(x)
        plt.ylabel(Ylabel, fontsize = Ylabel_fontsize, labelpad = 10)
        plt.xlabel(Xlabel, fontsize = Xlabel_fontsize, labelpad = 10)
        plt.locator_params(axis='x', nbins=5)  

        # ax1.yaxis.set_label_coords(-0.3, 0.5)
        ax1.xaxis.set_label_coords(0.45, -0.27)
        # ax1.set_ylim(bottom=20)

        # plt.show()
        # plt.tight_layout()
        pdfName = dataName[0:-4]
        # plt.savefig('./pdf/{}.eps'.format(pdfName)) 
        plt.savefig('./pdf/{}.jpg'.format(pdfName)) 
        # Xlabel = Xlabel.replace(' (%)','')
        # Xlabel = Xlabel.replace('# ','')
        # Ylabel = Ylabel.replace('# ','')
        plt.close()

    def genMultiName(self, multiple):
        if multiple == 0:
            return str()
        else:
            return "($" + "10" + "^{" + str(multiple) + "}" + "$)"
        
def get_n_index(sorted_list, n):
    if len(sorted_list) < 2 or n < 2:
        return sorted_list[:n]

    common_diff = sorted_list[1] - sorted_list[0]
    step_size = (sorted_list[-1] - sorted_list[0]) / (n - 1)

    result = []
    i = 0
    while len(result) < n:
        if i >= len(sorted_list):
            i = len(sorted_list) - 1
        # result.append(sorted_list[i])
        result.append(i)
        i += int(round(step_size / common_diff))
    # result.append(len(sorted_list) - 1)
    return result
if __name__ == "__main__":
    # data檔名 Y軸名稱 X軸名稱 Y軸要除多少(10的多少次方) Y軸起始座標 Y軸終止座標 Y軸座標間的間隔
    # ChartGenerator("numOfnodes_waitingTime.txt", "need #round", "#Request of a round", 0, 0, 25, 5)
    Xlabels = ["#RequestPerRound", "totalRequest", "#nodes", "r", "swapProbability", "alpha", "SocialNetworkDensity", "preSwapFraction", "entanglementLifetime"]
    Ylabels = ["algorithmRuntime", "waitingTime", "idleTime", "usedQubits", "temporaryRatio" , "entanglementPerRound" , "successfulRequest" , "usedLinks"]
    
    for Xlabel in Xlabels:
        for Ylabel in Ylabels:
            dataFileName = Xlabel + '_' + Ylabel + '.txt'
            ChartGenerator(dataFileName, Ylabel, Xlabel)


    Xlabel = "Timeslot"
    Ylabel = "#remainRequest"
    dataFileName = Xlabel + "_" + Ylabel + ".txt"
    ChartGenerator(dataFileName, Ylabel, Xlabel)

    Xlabel = "Timeslot"
    Ylabel = "#successRequest"
    dataFileName = Xlabel + "_" + Ylabel + ".txt"
    ChartGenerator(dataFileName, Ylabel, Xlabel)


    # Xlabel = "Timeslot"
    # Ylabel = "#entanglement"
    # dataFileName = Xlabel + "_" + Ylabel + ".txt"
    # ChartGenerator(dataFileName, Ylabel, Xlabel)