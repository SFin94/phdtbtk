import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


'''Script to print out the convergence information, opt trajectory and final energy for a job. Can optionally plot the optimisation trajectory if third argument included
    Usage: [gaussOpt.py] fileName freqeuncyNumber (t)

    Where:
        filename: name of the log file (str; with .log extension)
        freqeuncyNumber: number of lines (int) of freqeuncies to print out
        -t: Optional flag to include at the end if plot of optimisation trajectory wanted
'''

def printConvergence(convergenceResult, convergenceFlags):

    plot = False
    print('Convergence:')
    for cFlag in convergenceFlags:
        if convergenceResult[cFlag][0] == 'NO':
            print('\t' + cFlag + ': ' + convergenceResult[cFlag][0] + '\t' + 'Result: ' + convergenceResult[cFlag][1] + '\t' + 'Threshold: ' + convergenceResult[cFlag][2])
            plot = True
        else:
            print('\t' + cFlag + ': ' + convergenceResult[cFlag][0])
    return(plot)


def parseInfo(fileName, freqGoal=1):

    convStepCount = 0
    freqCount = 0
    jobInput = None

    # Set empty dicts for recording results
    convergenceFlags = ['Maximum Force', 'RMS     Force', 'Maximum Displacement', 'RMS     Displacement']
    optValues = {cFlag: [] for cFlag in convergenceFlags}
    optValues['Energy'] = []
    convResult = dict.fromkeys(convergenceFlags)
    convTol = dict.fromkeys(convergenceFlags)

    finalOutput = ''
    lowFreq = []

    # Track values of energy and convergence flag options
    for el in fileName:

        if ('#' in el) & (jobInput == None):
            jobInput = ''
            while el[1] != '-':
                jobInput += el.strip()
                el = fileName.__next__()
            print('Job input: ' + jobInput)

        if 'SCF Done' in el:
            optValues['Energy'].append(float(el.split('=')[1].split()[0]))

        for cFlag in convergenceFlags:
            if cFlag in el:
                optValues[cFlag].append(float(el.split()[2]))
                convResult[cFlag] = [el.split()[4], el.split()[2], el.split()[3]]
                convStepCount += 0.25

        if 'Low frequencies' in el:
            print(el.strip())

        if (('Frequencies' in el) & (freqCount < freqGoal)):
            print(el.strip())
            freqCount += 1

        if 'Link1:' in el:
            plot = printConvergence(convResult, convergenceFlags)

        if 'Error termination' in el:
            raise Exception('Job terminated with error')

    printConvergence(convResult, convergenceFlags)
    print('Final Energy: ' + str(optValues['Energy'][-1]))
    print('Steps taken to optimise: ' + str(convStepCount))
    return(optValues, convResult, plot)


def plotTraj(optValues, convResult):

    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Set figure size
    fig, ax = plt.subplots(figsize=(12,10))
    ax1 = ax.twinx()

    # Remove plot frame lines
    for axis in ax, ax1:
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(labelsize=12)

    # Set colours
    colour = sns.cubehelix_palette(4, start=1.5, dark=0, light=0.5)
    energyCol = sns.cubehelix_palette(4, start=.5, rot=-1.0, dark=0, light=0.5)[1]
    colInd = 0

    # Generate x axis
    stepsTaken = list(range(0, len(optValues['Energy'])))

    # Set axis labels and second axis
    fs = 13
    ax.set_xlabel('Step number', fontsize=fs)
    ax.set_ylabel('Energy (h)', fontsize=fs)
    ax1.set_ylabel('Opt quantity', fontsize=fs)

    for optFlag, optVal in optResults.items():
        if optFlag == 'Energy':
            ax.plot(stepsTaken, optVal, color=energyCol, alpha=0.6, lw=5, label='Energy')
        else:
            ax1.plot(stepsTaken, optVal, color=colour[colInd], label=optFlag, alpha=0.6, lw=3, ls='--')
            ax1.plot(stepsTaken, [float(convResult[optFlag][2])]*len(stepsTaken), color=colour[colInd], lw=0.5)
            colInd += 1
    ax1.legend(frameon=False)

    plt.show()


if __name__ == '__main__':

    usage = "usage: %(prog)s [fileName] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    # Set parser arguments and process
    parser.add_argument("fileName", nargs=1, type=argparse.FileType('r+'),  help="Gaussian optimisatin log file name")
    parser.add_argument("-f", "--freqs", dest='freqGoal', type=int, default=0, help="Number of real (i.e. Low frequencies omitted) vibrational modes to display")
    parser.add_argument("-t", "--traj", dest='traj', action='store_true',
                        help="Boolean flag which prints optimisation trajectory if True.")
    args = parser.parse_args()

    optResults, convResult, plot = parseInfo(args.fileName[0], args.freqGoal)

    if any([plot, args.traj]):
        plotTraj(optResults, convResult)


