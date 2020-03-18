import sys
import argparse
import numpy as np
import gaussGeom as gg


'''Script which can write a gaussian input file using CL arguments or a saved default

    Current job types available are:
        Opt + Freq (default), Opt, Freq, TS Opt, Relaxed scan (Opt(ModRedundant)), or own job types can be entered
    Method, basis set, SMD (water only), ModRedundant, computational resources and geometry input options can all be set.
'''


def parseOriginal(fileName, section):

    ''' Function which reads in the original .com file to pull out geometry or modRedundant input'''

    sectionCount = 0
    input = []
    with open(fileName + '.com', 'r') as inputFile:
        for el in inputFile:
            if el.strip() == '':
                sectionCount += 1
            elif sectionCount == section:
                input.append(el.strip())
    return(input)


if __name__ == "__main__":

    '''Argparse usage and arguments'''

    usage = "usage: %(prog)s [fileName] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("fileName", nargs=1, type=str, help="The input .com file without the .com extension")
    parser.add_argument("-j", "--job", dest="jobType", nargs='*', type=str, default=['opt', 'freq'],
                        help="Gaussian job type, currently available: opt, freq, reopt (opt, freq from chk), scan (opt(ModRedundant)), ts (TS Opt), own (enter own arguments)")
    parser.add_argument("-g", "--geom", dest="geom", nargs='*', type=str, default='file')
    parser.add_argument("-m", "--method", dest="method", type=str, default='M062X',
                        help="The method to be used, give the correct gaussian keyword")
    parser.add_argument("-b", "--basis", dest="basisSet", type=str, default='6-311++G(d,p)',
                        help="The basis set to be used, give the correct gaussian keyword")
    parser.add_argument("--mod", dest="modRed", type=str,
                        help="ModRedundant input, each input line entered as a csv")
    parser.add_argument("--smd", dest="smd", action='store_true',
                        help="Flag whether to include SMD keyword for solvation or not, set to true for input")
    parser.add_argument("-p", dest="preset", nargs=1, type=int,
                        help="Preset flag to set required prcoessors and mem")

    args = parser.parse_args()


    # Sets filename and removes '.com' if present at the end of the name
    fileName = args.fileName[0]
    if fileName[-4:] == '.com':
        fileName = fileName[:-4]

    # Set the job method
    jobMethod = args.method+'/'+args.basisSet

    # Set default values for nProc and memMB assuming opt and freq as default
    nProc = 20
    memMB = 60000

    # Set the job type and job title
    jobType = ''
    jobTitle = fileName
    for jT in args.jobType:
        if jT.lower() == 'opt':
            if args.modRed == True:
                jobType += 'Opt(Tight,ModRedundant) '
            else: jobType += 'Opt(Tight) '
        if jT.lower() == 'reopt':
            jobType += 'Opt(Tight,RCFC) Freq '
        if jT.lower() == 'freq':
            jobType += 'Freq '
        if jT.lower() == 'ts':
            if args.geom == 'chk':
                jobType += 'Opt(Tight,TS,NoEigen,RCFC) Freq '
            else:
                jobType += 'Opt(Tight,TS,NoEigen,CalcFC) Freq '
        if jT.lower() == 'scan':
            jobType += 'Opt(ModRedundant,MaxCycles=100) '
            try:
                args.modRed != None
            except:
                print("ModRedundant input expected for scan but not given", sys.stderr)
            nProc = 40
            memMB = 120000
        if jT.lower() == 'own':
            jobType += input("Enter job inputs as they would appear in Gaussian.com file:\n") + ' '
        jobTitle = jobTitle + ' ' + jT

    # Adds SCRF command and SMD for solvation in water if flag raised at input
    if args.smd == True:
        jobType += 'SCRF(SMD) '
    # Set the job Spec up with standard convergence criteria
    jobSpec = '#P ' + jobMethod + ' ' + jobType + 'SCF(Conver=9) Int(Grid=UltraFine)'


    # Sets charges + multiplicity and/or molecular geometry from original file
    if args.geom == 'file':
        moleculeGeom = parseOriginal(fileName, 2)
    elif args.geom == 'chk':
        jobSpec += ' Geom(Check) Guess(Read)'
        moleculeGeom = [parseOriginal(fileName, 2)[0]]
    elif args.geom == 'allchk':
        jobSpec += ' Geom(AllCheck) Guess(Read)'
    elif args.geom[0][-4:] == '.log':
        geomLog = args.geom[0]
        with open(geomLog, 'r') as logFile:
            for el in logFile:
                if ('Charge' in el) and ('Multiplicity' in el):
                    moleculeGeom = [el.split()[2] + ' ' + el.split()[-1]]
        ids = gg.atomIdentify(geomLog)
        if len(args.geom) == 2:
            optStep = int(args.geom[-1])
        else:
            optStep = 1
        geometry = gg.geomPulllog(geomLog, optStep=int(optStep))[0]
        for atom in range(len(ids)):
            moleculeGeom.append('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(ids[atom], geometry[atom,:]))


    # Sets modredundant input from next section or user input
    if args.modRed != None:
        modRedundant = args.modRed.split(',')

    # Parses in presets and sets computational variables from that
    if args.preset != None:
        try:
            with open('/Volumes/home/bin/.presets', 'r') as presets:
                presetNum = 0
                for el in presets:
                    if el[0] != '#':
                        presetNum += 1
                    if presetNum == (args.preset[0]+1):
                        nProc = int(el.split(';')[2])
                        memMB = int((el.split(';')[3])[:-2]) - nProc*100
        except IOError:
            print("Couldn't locate the presets file in '~/bin/.presets'", sys.stderr)

    # Writes new .com file
    with open(fileName+'.com', 'w+') as output:
        print('%chk={}'.format(fileName), file=output)
        print('%nprocshared={:d}'.format(nProc), file=output)
        print('%mem={:d}MB'.format(memMB), file=output)
        print(jobSpec + '\n', file=output)
        if args.geom != 'allchk':
            print(jobTitle + '\n', file=output)
            for el in moleculeGeom:
                print(el, file=output)
        if args.modRed != None:
            print('', file=output)
            for el in modRedundant:
                print(el, file=output)
        print('\n\n', file=output)




