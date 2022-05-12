import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
from Plotter import dataset, figret
import Plotter as PL

class inputRange:
    def __init__(self,data=None,dmin=None,dmax=None,name=None):
        self.data = data
        self.dmin = dmin
        self.dmax = dmax
        self.name= name


class explorerPlot:

    def __init__(self,funcs,x_lin,drs:[inputRange]):
        #func parameters should match order of drs
        #func should also be able to handle n args
        self.update_functions = funcs
        self.dataranges = drs
        self.xlin = x_lin
        self.figret = None
        self.reset_button = None
        self.slider_array = []
        self.plot_x_bound=None
        self.plot_y_bound=None
        self.line_x_scale='linear'
        self.line_y_scale='linear'
        self.update_bounds = True
        self.slider_width = 0.2


    def buildPlot(self):
        input_data = []
        for id in self.dataranges:
           input_data.append(id.data)
        ds = dataset()
        ds.x = self.xlin
        ds.y = self.update_functions(*input_data)
        ds.plot_type = ds.plottype
        # create plot
        pl = PL.Plotter.Plotter()
        if(self.plot_y_bound==None):
            ymax=1.2*max(ds.y)
            ymin=0.8*min(ds.y)
        else:
            ymax=self.plot_y_bound[0]
            ymin = self.plot_y_bound[1]
        if (self.plot_x_bound == None):
            xmax =max(ds.x)
            xmin = min(ds.x)
        else:
            xmax = self.plot_x_bound[0]
            xmin = self.plot_x_bound[1]
        fr = pl.Plot([ds], xscale=self.line_x_scale, yscale=self.line_y_scale,maxy=ymax,miny=ymin,minx=xmin,maxx=xmax)
        self.figret = fr
        self.buildSliders()

    def updateBounds(self,maxy,miny):
        self.figret.ax.set_ylim(miny,maxy)


    def buildSliders(self):
        axcolor = 'lightgoldenrodyellow'
        dr_len  = len(self.dataranges)
        height_adjust = self.slider_width*self.figret.ax.get_position().height
        wad = height_adjust/(dr_len+1)
        self.figret.pyplt.subplots_adjust(bottom=height_adjust)
        for j in range(dr_len):
            df = self.dataranges[j]
            ax = self.figret.pyplt.axes([self.figret.ax.get_position().x0, self.figret.ax.get_position().y0*0.76 - (wad*(j+1)/1.5), self.figret.ax.get_position().width,wad], facecolor=axcolor)
            step=(df.dmax- df.dmin)/100
            slider = Slider(ax, df.name, df.dmin, df.dmax, valinit=df.data, valstep=step)
            slider.on_changed(self.update)
            self.slider_array.append(slider)
        self.buildResetButton()

    def buildResetButton(self):
        resetax = self.figret.pyplt.axes([self.figret.ax.get_position().x0+self.figret.ax.get_position().width, self.figret.ax.get_position().y0, 0.08, 0.04])
        self.reset_button = Button(resetax, 'Reset', hovercolor='0.975')
        self.reset_button.on_clicked(self.reset)

    def update(self,val):
        input_data = []
        yval = None
        for s in self.slider_array:
            input_data.append(s.val)
        yval = self.update_functions(*input_data)
        if(self.plot_y_bound==None and self.plot_x_bound == None and self.update_bounds):
            ymax = 1.2 * max(yval)
            ymin = 0.8 * min(yval)
            self.updateBounds(maxy=ymax,miny=ymin)
        self.figret.plots[0][0].set_ydata(yval)
        self.figret.fig.canvas.draw_idle()

    def reset(self,event):
        for s in self.slider_array:
            s.reset()





