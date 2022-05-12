import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from Plotter import dataset, figret

class inputRange:
    def __init__(self,data=None,dmin=None,dmax=None,name=None):
        self.data = data
        self.dmin = dmin
        self.dmax = dmax
        self.name= name

class explorerPlot:

    def __init__(self,funcs,drs:[inputRange]):
        #func parameters should match order of drs
        #func should also be able to handle n args
        self.buildSliders()
        self.update_functions = funcs
        self.dataranges = drs



    def buildPlot(self):
        # parameter space
        a = inputRange(0.1, 0, 1, "a")
        w = inputRange(0.1, 0, 2, "w")
        phi = inputRange(0.1, 0, 1, "$$\Phi$$")
        # package parameter pace
        fargs = [a, w, phi]
        fargt = tuple(fargs)
        # first results
        ds = dataset()
        ds.x = x
        ds.y = list(sinwave(fargs[0].data, fargs[1].data, fargs[2].data))
        ds.plot_type = ds.plottype
        # create plot
        pl = PL.Plotter()
        fr = pl.Plot([ds], xscale="linear", yscale="linear")
        return fr

    def buildSliders(self):
        axcolor = 'lightgoldenrodyellow'

        for j in range(len(self.dataform)):
            df = self.dataform[j]
            ax = self.figret.pyplt.axes([0.15, 0.40 - (0.03*(j+1)), 0.65, 0.03], facecolor=axcolor)
            step=(df.max- df.min)/100
            slider = Slider(ax, df.name, df.min, df.max, valinit=df.data, valstep=step)
            slider.on_changed(self.updateSlider)
            self.slider_array.append(slider)


    def updateSlider(self):
        #update fargs
        for s in self.slider_array:

        #getnew results
        #tie ds in dss to f in funcs
        results = []
        for f in self.update_functions:
            results.append(f())
