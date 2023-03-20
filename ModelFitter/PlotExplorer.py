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
        self.maxy =None
        self.miny = None
        self.update_bounds = True
        self.slider_width = 0.2
        self.datasetind=0
        self.dataset=[]


    def buildPlot(self):
        input_data = []
        for id in self.dataranges:
           input_data.append(id.data)
        # create plot
        pl = PL.Plotter.Plotter()
        ret = self.update_functions(*input_data)
        if (self.xlin is None):
            self.xlin = ret[0]
            ret = [ret[1]]
        try:
            len(ret)
        except:
            ret = np.array(ret)
        for i in range(len(ret)):
            ds = dataset()
            ds.x = self.xlin
            ds.y = ret[i]
            ds.plot_type = ds.plottype
            self.dataset.append(ds)
            if(self.maxy is None):
                self.maxy = max(ds.y)
            elif(self.maxy <max(ds.y)):
                self.maxy  = ds.y
            if (self.miny  is None):
                self.miny  = min(ds.y)
            elif (self.miny  > min(ds.y)):
                self.miny  = ds.y

        if(self.plot_y_bound==None):
            ymax=1.2*self.maxy
            ymin=0.8*self.miny
        else:
            ymax=self.plot_y_bound[0]
            ymin = self.plot_y_bound[1]
        if (self.plot_x_bound == None):
            xmax =max(self.xlin)
            xmin = min(self.xlin)
        else:
            xmax = self.plot_x_bound[0]
            xmin = self.plot_x_bound[1]

        fr = pl.Plot(self.dataset, xscale=self.line_x_scale, yscale=self.line_y_scale,maxy=ymax,miny=ymin,minx=xmin,maxx=xmax)
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
        self.dataset = []
        yval = None
        for s in self.slider_array:
            input_data.append(s.val)
        ret = self.update_functions(*input_data)
        try:
            len(ret)
            if(len(ret)>1):
                if (self.xlin is None or len(ret[0]) == len(ret[1])):
                    self.xlin = ret[0]
                    ret = [ret[1]]
        except:
            None

        try:
            len(ret)
        except:
            ret = np.array(ret)
        j = None
        for i in range(len(ret)):
            ds = dataset()
            ds.x = self.xlin
            ds.y = ret[i]
            ds.plot_type = ds.plottype
            self.dataset.append(ds)
            # if (self.maxy is None):
            #     self.maxy = max(ds.y)
            # elif (self.maxy < max(ds.y)):
            #     self.maxy = ds.y
            # if (self.miny is None):
            #     self.miny = min(ds.y)
            # elif (self.miny > min(ds.y)):
            #     self.miny = ds.y
            #[0][0] first line
            if(j is None):
                j=i
            else:
                j+=1
            while (type(self.figret.plots[j]) is not list):
                j += 1
            # self.figret.plots[j][0].set_ydata(ret[i])
            self.figret.plots[j][0].set_data(self.xlin,ret[i])

        # if(self.plot_y_bound==None and self.plot_x_bound == None and self.update_bounds):
        #     ymax = 1.2 * self.maxy
        #     ymin = 0.8 * self.miny
        #     self.updateBounds(maxy=ymax,miny=ymin)

        self.figret.fig.canvas.draw_idle()

    def reset(self,event):
        for s in self.slider_array:
            s.reset()





