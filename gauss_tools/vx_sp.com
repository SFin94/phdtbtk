%chk=vx_sp
%nprocshared=20
%mem=60000MB
#P M062X/6-311++G(d,p) Opt(Tight,RCFC) Freq SCRF(SMD) SCF(Conver=9) Int(Grid=UltraFine) Geom(Check) Guess(Read)

vx_sp reopt

0 1



