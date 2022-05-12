import numpy as np
import ModelFitter as MF
from ModelFitter import dataform as df
from ModelFitter import constraints as con
from Plotter.Plotter import dataset,figret
import Plotter.Plotter as PL

x = np.linspace(0,2*np.pi)
x = list(x)
def sinwave(a,w,phi):
    f = a*np.sin(np.array(x)*w + phi)
    return f

def getcompare():
    a=0.7
    w=1.5
    phi=0.9
    f = sinwave(a,w,phi)
    ds = dataset()
    ds.x=x
    ds.y=list(f)
    tx = np.linspace(0,1,len(x))
    ds.x_error = list(tx)
    ds.y_error = list(tx**8)
    ds.plot_type = ds.scattertype
    return ds
def getxandy(a,w,phi):
    f=sinwave(a=a,w=w,phi=phi)          
    ds = dataset()
    ds.x=x
    ds.y=list(f)
    return ds
def confun(*args):
    return args[0][0]
def run():
    #parameter space
    acon = con(confun, 0.8,1.5,0.5)
    a = df(0.1,0,1,1,"a",[acon])
    w = df(0.1,0,2,1,"w")
    phi = df(0.1,0,1,1,"phi")
    #package parameter pace
    fargs = [a,w,phi]
    fargt = tuple(fargs)
    #first results
    ds = dataset()
    ds.x = x
    ds.y = list(sinwave(fargs[0].data,fargs[1].data,fargs[2].data))
    ds.plot_type=ds.plottype
    #create plot
    pl = PL.Plotter()
    fr = pl.Plot([ds],xscale="linear",yscale="linear")
    #feed fitter
    mf = MF.Fitter(func=getxandy,ds=getcompare(),figret=fr,fargs=fargt,grad=False,show_error=True,manual_mode=False)
    mf.par_opt_len = 7
    mf.maxinter = 400
    mf.random_search=True
    mf.run()
    # mf.errorplot.fig.savefig("test")
    # mf.errorSlider.val = 1
    # mf.updateErrorPlot(None)
    # mf.errorplot.fig.savefig("test3")
    # mf.figret.fig.savefig("test2")

run()