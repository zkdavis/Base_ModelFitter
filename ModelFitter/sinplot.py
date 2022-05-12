import PlotExplorer as PE
import numpy as np

x = np.linspace(0,2*np.pi)
x = list(x)
def sinwave(a,w,phi):
    f = a*np.sin(np.array(x)*w + phi)
    return f

def run():
    #parameter space
    ir = PE.inputRange
    a = ir(0.1,0,1,"a")
    w = ir(0.1,0,2,"w")
    phi = ir(0.1,0,1,"phi")
    #package parameter pace
    fargs = [a,w,phi]
    fargt = tuple(fargs)
    ep = PE.explorerPlot(sinwave,x,fargs)
    ep.buildPlot()
    ep.figret.pyplt.show()

run()