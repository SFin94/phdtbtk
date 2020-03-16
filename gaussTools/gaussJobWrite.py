import sys
import argparse
import numpy as np
import gaussGeom as gg

# Script which write the input file for a gaussian calculation
# Want input arguments of the job type
# Modredundant output for the scan

# Can pull method etc from previous file
# inputs to job type are not case sensitive

# Read in original .com input file if geometry is present
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

usage = "usage: %(prog)s [fileName] [args]"
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("fileName", nargs=1, type=str, help="The input .com file without the .com extension")
parser.add_argument("-j", "--job", dest="jobType", nargs='*', type=str, default=['opt', 'freq'])
parser.add_argument("-g", "--geom", dest="geom", type=str, default='file')
parser.add_argument("-m", "--method", dest="method", nargs=1, type=str, default=['M062X'],
                    help="The method to be used, give the correct gaussian keyword")
parser.add_argument("-b", "--basis", dest="basisSet", nargs=1, type=str, default=['6-311++G(d,p)'],
                    help="The basis set to be used, give the correct gaussian keyword")
parser.add_argument("--mr", "--mod", dest="modRed", action='store_true',
                    help="Flag whether to expect moderedundant input or not, set to true for input")
parser.add_argument("--smd", dest="smd", action='store_true',
                    help="Flag whether to include SMD keyword for solvation or not, set to true for input")
parser.add_argument("-p", dest="preset", nargs=1, type=int,
                    help="Preset flag to set required prcoessors and mem")

args = parser.parse_args()


# Set jobspec defaults


# Sets filename and removes '.com' if present at the end of the name
fileName = args.fileName[0]
if fileName[-4:] == '.com':
    fileName = fileName[:-4]

# Set the job method and type keywords for the new input file
jobMethod = args.method[0]+'/'+args.basisSet[0]

# Set default values for nProc and memMB assuming opt and freq as default
nProc = 20
memMB = 60000

# Set the job type and job title

'''
    Current default job types available are:
        Opt, Freq (default)
        Opt (w/woutModRedundant flag set)
        Freq
        TS: Opt(TS,NoEigen,RCFC) Freq
        Scan: Opt(ModRedundant,MaxCycles=100), sets ModRedundant flag to True and default proc and mem
        Own: Own input (str)

        should be able to build (o, f) then other options
    '''

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
        args.modRed = True
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
elif args.geom[-4:] == '.log':
    with open(args.geom, 'r') as logFile:
        for el in logFile:
            if ('Charge' in el) and ('Multiplicity' in el):
                moleculeGeom = [el.split()[2] + ' ' + el.split()[-1]]
    ids = gg.atomIdentify(args.geom)
    optStep = input("If scan log file enter which optimised geometry to pull (press enter to skip)")
    if optStep == '':
        optStep = 1
    geometry = gg.geomPulllog(args.geom, optStep=int(optStep))[0]
    for atom in range(len(ids)):
        moleculeGeom.append('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(ids[atom], geometry[atom,:]))

# Sets modredundant input from next section or user input
if args.modRed == True:
    # Edit to check file for modRedundant input then if not there add user input - but issue with connectivity changing section value
    #modRedundant = parseOriginal(fileName, 2)
    modRedundant = input("Enter modRedundant input (csv for multiple lines):").split(',')

# Parses in presets and sets variables from that?
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
    if args.modRed == True:
        print('', file=output)
        for el in modRedundant:
            print(el, file=output)
    print('\n\n', file=output)




