import PlotExplorer as PE
import numpy as np
import Plotter as PL

x = np.linspace(0,2*np.pi)
x = list(x)
def sinwave(a,w,phi):
    f = a*np.sin(np.array(x)*w + phi)
    f2 = a*np.cos(np.array(x)*w + phi)
    return [f,f2]

def run():
    #parameter space
    ir = PE.inputRange
    a = ir(0.1,0,1,"a")
    w = ir(0.1,0,2,"w")
    phi = ir(0.1,0,1,"phi")
    #package parameter pace
    fargs = [a,w,phi]
    ep = PE.explorerPlot(sinwave,x,fargs)
    ds = PL.dataset()
    xx  = x
    ds.x =xx
    ds.y = xx
    ds.plot_type = ds.scattertype
    ep.dataset=[ds]
    ep.datasetind = len(ep.dataset)
    ep.buildPlot()
    ep.figret.pyplt.show()

run()