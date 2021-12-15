import numpy as np
from matplotlib import animation
from matplotlib.widgets import Button, Slider
import matplotlib.pyplot as plt
from Plotter import dataset, figret
import math,random


class constraints:
    def __init__(self,f,val,max,min):
        self.f = f
        self.max = max
        self.min=min
        self.val = val
class dataform:
    def __init__(self,data,min,max,lrate=None,name=None,cons=None):
        self.data=data
        self.min=min
        self.max=max
        self.lrate=lrate
        self.name=name
        self.constraints=cons


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def find_nearest_point(arrayy,arrayx,x,y):
    arrayy = np.asarray(arrayy)
    arrayx = np.asarray(arrayx)
    newx=None
    newy=None
    for i in range(len(arrayy)):
        if(newx==None):
            newx=arrayx[i]
            newy=arrayy[i]
        else:
            r=np.abs(((newx**2) + (newy**2)) - (x**2 + y**2))
            rp = np.abs((arrayx[i]**2) + (arrayy[i]**2) - (x**2 + y**2))
            if( rp<r):
                newx = arrayx[i]
                newy = arrayy[i]
    return newx,newy

def getGradient(y: [], x: [], i: int = None, minchange: float = 1e-4, delfactor: float = 0.5):
    a = None
    b = None
    delf = None
    if (i == None):
        a = (y[-1] - y[-2])
        b = (x[-1] - x[-2])
    else:
        a = (y[i] - y[i - 1])
        b = (x[i] - x[i - 1])
    #todo add second order functionality


    # if it is not greater than minchange  in x it be a factor of the y
    if (np.abs(b) >= minchange and np.abs(a)>0):
        grad = (b) / (a)
    else:
        rand = np.random.rand()
        grad = x[-1]*((rand-0.5))*5e-1
    # if (np.abs(grad) == np.nan or np.abs(grad) < minchange):
    #     grad = delfactor * a
    if(math.isnan(grad)):
        print("asfd")

    return grad


class Fitter:
    def __init__(self, func, ds: dataset, figret: figret, fargs:[dataform],max_int:int=10000,show_error:bool=True,grad:bool=True,manual_mode:bool=False):
        tem = []
        for i in fargs:
            tem.append(i.data)
        self.iteration=0
        self.dataform=fargs
        self.args = tuple(tem)
        self.ds = ds
        self.func = func
        self.figret = figret
        self.errors = []
        self.grad=grad
        self.all_error=[]
        self.reset_count = 0
        self.resetmax=10
        self.slider_changed=False
        self.pkfound = False
        self.error_count=0
        self.oldargs = []
        self.oldargs_byparameter=[]
        self.slopes_byparameter=[]
        self.slope_signchangecount=[]
        self.slope_signchangecountmax=6
        self.cur_par = 0
        self.cur_par_count = 0
        self.par_opt_len = 6
        self.learning_rate = 0.001
        self.org_learning_rate = 0.001
        self.close_error=2
        self.chi=-1
        self.lcount=0
        self.maxlcount=3
        self.figret.ax.errorbar(ds.x, ds.y, yerr=ds.y_error, xerr=ds.x_error, c="C1", fmt='o',ms=3)
        self.maxinter=max_int+1
        self.errorplot=None
        self.show_error=show_error
        self.errorSlider=None
        self.random_search=True
        self.complete=False
        self.updating_sliders = False
        self.cur_err_par=0
        self.pause=False
        self.errorBetterCount=0
        self.errorBetterCountMax=10
        self.manual_mode = manual_mode
        if(self.manual_mode != True):
            self.ani = animation.FuncAnimation(figret.fig, self.update, interval=10, blit=False, repeat=False,
                                          frames=self.maxinter)
        axnext = self.figret.pyplt.axes([0.86, 0.25, 0.1, 0.075])
        self.bpause = Button(axnext, 'Play/Pause')
        axshowbesst = self.figret.pyplt.axes([0.86, 0.25 + 0.075, 0.1, 0.075])
        self.showbest = Button(axshowbesst, 'Show Best')
        self.slider_array = []
        self.figret.pyplt.subplots_adjust(left=0.25, bottom=0.5)
        self.buildSliders()
        # self.figret.fig.canvas.mpl_connect('button_press_event', self.playpause)
        self.bpause.on_clicked(self.playpause)
        self.showbest.on_clicked(self.showBest)


    def run(self):
        self.figret.pyplt.show()
    def buildSliders(self):
        axcolor = 'lightgoldenrodyellow'

        for j in range(len(self.dataform)):
            df = self.dataform[j]
            ax = self.figret.pyplt.axes([0.15, 0.40 - (0.03*(j+1)), 0.65, 0.03], facecolor=axcolor)
            step=(df.max- df.min)/100
            slider = Slider(ax, df.name, df.min, df.max, valinit=df.data, valstep=step)
            slider.on_changed(self.updateSlider)
            self.slider_array.append(slider)

    def updateSlider(self,event):

        self.slider_changed=True
        if(self.manual_mode == True and self.updating_sliders == False):
            self.update(self.iteration)
            self.reset_count = 1

    def showBest(self,event):
        if(self.manual_mode != True):
            if (self.pause == False):
                self.playpause(event)
            minarg = np.array(self.all_error).argmin()
            self.arg = self.oldargs[minarg]
            newds = self.func(*self.arg)
            plots = self.figret.plots
            tplots = plots[0]
            # updaating sliders
            for j in range(len(self.slider_array)):
                self.slider_array[j].set_val(self.arg[j])
            try:
                if (len(tplots) > 0):
                    tplots[0].set_ydata(newds.y)
                else:
                    plots[0].set_ydata(newds.y)
            except Exception:
                plots[0].set_ydata(newds.y)
        elif(self.manual_mode == True):
            self.run_finished()
            minarg = np.array(self.all_error).argmin()
            self.arg = self.oldargs[minarg]
            newds = self.func(*self.arg)
            plots = self.figret.plots
            tplots = plots[0]
            # updaating sliders
            for j in range(len(self.slider_array)):
                self.slider_array[j].set_val(self.arg[j])
            try:
                if (len(tplots) > 0):
                    tplots[0].set_ydata(newds.y)
                else:
                    plots[0].set_ydata(newds.y)
            except Exception:
                plots[0].set_ydata(newds.y)

    def playpause(self,event):
        if(self.manual_mode != True):
            if(self.pause==False):
                self.ani.event_source.stop()
                self.pause=True
            else:
                self.ani.event_source.start()
                self.pause=False
        else:
            minarg = np.array(self.all_error).argmin()
            self.arg = self.oldargs[minarg]
            newds = self.func(*self.arg)
            plots = self.figret.plots
            tplots = plots[0]
            # updaating sliders
            for j in range(len(self.slider_array)):
                self.slider_array[j].set_val(self.arg[j])
            try:
                if (len(tplots) > 0):
                    tplots[0].set_ydata(newds.y)
                else:
                    plots[0].set_ydata(newds.y)
            except Exception:
                plots[0].set_ydata(newds.y)

    def update(self, i):
        self.iteration = i
        f = self.func
        args = self.args
        if(self.slider_changed):
            self.slider_changed=False
            temp=[]
            for j in self.slider_array:
                temp.append(j.val)
            args=tuple(temp)
            # updaating sliders
            self.args = args
        #check for constraints
        if(self.manual_mode != True):
            for df in self.dataform:
                dfcon = df.constraints
                if(dfcon is not None):
                    for tc in dfcon:
                        tf = tc.f
                        tmax = tc.max
                        tmin = tc.min
                        targ = self.args
                        tv = tf(targ)
                        maxrecur =9000
                        count =0
                        while(tv>tmax or tv<tmin or count>maxrecur):
                            print("constraint hit")
                            print(count)
                            self.args = self.startNewPar(self.cur_par, curParam=False, completelyrandom=True)
                            targ = self.args
                            tv = tf(targ)
                            count +=1
                        # if (self.iteration >= 1):
                        #     self.iteration -= 1
                        # else:
                        #     self.iteration = 0



        #gets newds with this iterations parameters
        newds = f(*args)

        #get error of newrun
        er = self.errorCalc(newds)

        #will use to update blueline later
        update_blue=False
        if(len(self.all_error)>1):
            miner = min(self.all_error)
            if(er<miner):
                update_blue=True
        # adds to current parameters errors
        try:
            self.errors[self.cur_par].append(er)
        except Exception as e:
            # extends errors to hold the current parameters.
            # todo chang to if
            self.errors.append([er])
        if(self.grad==False):
            # adds history of old arguments
            try:
                self.oldargs_byparameter[self.cur_par].append(self.args[self.cur_par])
            except Exception as e:
                # extends list of arguments
                # todo change to if
                self.oldargs_byparameter.append([self.args[self.cur_par]])
        # adds to all error
        self.all_error.append(er)
        # adds args as tuple to history
        self.oldargs.append(self.args)
        if(self.manual_mode != True):
            if(self.grad == False):
                if (self.cur_par_count == 0):
                    if (len(self.all_error) > self.par_opt_len):
                        if (self.all_error[-1] > min(self.all_error[-self.par_opt_len:-1])):
                            tt = np.argmin(self.all_error[-self.par_opt_len:-1])
                            self.args = list(self.args)
                            self.args[self.cur_par] = self.oldargs[-(self.par_opt_len - tt)][self.cur_par]
                            self.args = tuple(self.args)
                            print("reverting set and slowing down previous")
                            cp = self.cur_par
                            cp = cp-1
                            if(cp<0):
                                cp=len(self.args)-1
                            if(self.dataform[cp].lrate>1e-3):
                                self.dataform[cp].lrate = self.dataform[cp].lrate * 0.5
                            self.cur_par_count+=1
                            return

        if (self.manual_mode != True):
            if(self.grad==False):
                if(len(self.errors) == len(self.args)):
                    if(len(self.errors[self.cur_par])>5):
                        if(self.all_error[-1]>self.close_error):
                            if(np.abs((self.errors[self.cur_par][-1] - self.errors[self.cur_par][-2])/self.errors[self.cur_par][-1])<1e-1):
                                if(self.dataform[self.cur_par].lrate<1e3):
                                    self.dataform[self.cur_par].lrate =self.dataform[self.cur_par].lrate*2
                                    print("speed increased")
                        if (self.random_search):
                            if(self.reset_count==0):
                                if(np.abs((self.all_error[-1] - min(self.all_error))/min(self.all_error))>3):
                                    self.args = self.oldargs[-2]
                                    self.reset_count+=1
                                    print("reverting random")
                                    self.cur_par_count += 1
                                    return

                        if (np.abs((self.all_error[-1] - min(self.all_error))/min(self.all_error)) >25):
                            minar = np.argmin(self.all_error)
                            self.args=self.oldargs[minar]
                            print("reverting terrible")
                            self.cur_par_count += 1
                            return


        #Plots error
        if(self.show_error and len(self.errors)== len(self.args)):
            self.plotError(i)

        newargs = list(args)
        if (self.manual_mode != True):
            #need at least two runs to start the process
            if (self.errors!=None and len(self.errors[self.cur_par])>=2):
                if(self.grad):
                    for k in range(len(args)):
                        self.update_param(k,newargs)
                else:

                    newargs=list(args)
                    newargs = self.update_param(self.cur_par,newargs)

            else:
                #start random if we don't have enough to get a gradient
                newargs = list(self.startNewPar(self.cur_par,curParam=True,center=True))

            if(self.grad == False):
                #checks if we need to move to a new parameter
                if (self.cur_par_count >= self.par_opt_len):
                    self.cur_par += 1
                    self.cur_par_count = 0
                    if (self.cur_par > (len(args) - 1)):
                        self.cur_par = 0

        ####updating plot
        self.args = newargs
        if(update_blue):
            plots = self.figret.plots
            tplots = plots[0]
            try:
                if(len(tplots)>0):
                    tplots[0].set_ydata(newds.y)
                else:
                    plots[0].set_ydata(newds.y)
            except Exception:
                plots[0].set_ydata(newds.y)
        self.figret.ax.set_title("Iterataion: " + str(i) + " Current Parameter: " + str(self.cur_par) +" cur_par_count: "+str(self.cur_par_count)+ " Chi: " + str("{:.4f}".format(self.chi) ))

        #updaating sliders
        self.updating_sliders=True
        for j in range(len(self.slider_array)):
            self.slider_array[j].set_val(self.args[j])
            self.updating_sliders = True
        self.updating_sliders = False
        self.slider_changed=False

        #when finished
        if(i>=self.maxinter-3):
            print("run finished called")
            self.run_finished()


    def update_param(self,param,newargs):
        done=False
        close=False
        k=param
        args = self.args
        tw = np.matrix(self.oldargs)
        allerr = self.all_error
        tw1 = tw[:, k]
        tw2 = []
        for j in tw1:
            tw2.append(float(j[0]))
        ter = self.all_error
        if(min(ter)<=self.close_error):
            close=True
        grad = getGradient(y=ter, x=tw2)

        try:
            self.slopes_byparameter[k].append(grad)
        except Exception as e:
            # extends list of arguments
            # todo change to if
            self.slopes_byparameter.append([grad])
        gfactor = grad * (self.dataform[k].lrate)
        diff = self.dataform[k].max - self.dataform[k].min
        if (np.abs(gfactor) > np.abs(diff / 10)):
            gfactor = (gfactor / np.abs(gfactor)) * diff / 10
        a = args[k] - gfactor
        #todo make add_min param
        add_min = 1e-4
        if(self.dataform[k].min != 0):
            add_min = self.dataform[k].min /100
        if (a <= self.dataform[k].min):

            a = self.dataform[k].min + add_min
            if(self.dataform[self.cur_par].lrate<1e-3):
                self.dataform[self.cur_par].lrate = self.dataform[self.cur_par].lrate * 0.5
                print("barrier slowed")
        elif (a >= self.dataform[k].max):
            a = self.dataform[k].max - add_min
            if (self.dataform[self.cur_par].lrate < 1e-3):
                self.dataform[self.cur_par].lrate = self.dataform[self.cur_par].lrate * 0.5
                print("barrier slowed")
        if(close):
            newargs[k] = a
        if (len(self.slopes_byparameter) == len(args)):
            for zz in range(len(args)):

                if (len(self.slopes_byparameter[zz]) > 3):
                    s1 = self.slopes_byparameter[zz][-1]
                    s2 = self.slopes_byparameter[zz][-2]
                    if (np.abs(s1) != 0):
                        sign1 = s1 / np.abs(s1)
                    else:
                        sign1 = 0
                    if (np.abs(s2) != 0):
                        sign2 = s2 / np.abs(s2)
                    else:
                        sign2 = 0
                    if (len(self.slope_signchangecount) == zz):
                        self.slope_signchangecount.append(0)
                    if (sign1 != sign2 and sign1 != 0 and sign2 != 0):
                        try:
                            self.slope_signchangecount[zz] += 1
                        except Exception as e:
                            # extends errors to hold the current parameters.
                            # todo chang to if
                            self.slope_signchangecount.append(1)
                    else:
                        try:
                            self.slope_signchangecount[zz] = 0
                        except Exception as e:
                            # extends errors to hold the current parameters.
                            # todo chang to if
                            self.slope_signchangecount.append(0)
                    while (len(self.slope_signchangecount) < zz + 1):
                        self.slope_signchangecount.append(0)
                    if(len(self.errors[self.cur_par])>3):
                        if (self.slope_signchangecount[zz] > self.slope_signchangecountmax and min(self.all_error)< 10*self.close_error):
                            if(self.dataform[zz].lrate > 1e-2):
                                self.slope_signchangecount[zz] = 0
                                self.dataform[zz].lrate = self.dataform[zz].lrate * 0.5
                                print("slowed")

        if(self.random_search):
            if (len(tw2) > 2):
                if (np.abs(tw2[-1] - tw2[-2]) < 1e-2 and np.abs(self.all_error[-1] - self.all_error[-2]) < 1e-2 and self.all_error[-1] < 10 and done == False):
                    if (self.reset_count < self.resetmax):
                        self.reset_count += 1
                    else:
                        self.reset_count = 0
                        a = self.startNewPar(curpar=k)[k]
                        # if (self.dataform[k].lrate < 1e2):
                        #     self.dataform[k].lrate = self.dataform[k].lrate * 1.01
                        newargs[k] = a
                        self.cur_par_count += 1
                        print("newparam b " + str(self.iteration))
                        done = True

        if (self.all_error[-1] > min(self.all_error) * 10 and done == False and close==False):
            if (self.reset_count < self.resetmax):
                self.reset_count += 1
            else:
                self.reset_count = 0
                cen = False
                if (self.all_error[-1] > 10):
                    cen = True
                a = self.startNewPar(curpar=k, center=cen)[k]
                self.cur_par_count += 1
                print("newparam a "+str(self.iteration))
                newargs[k] = a
                done=True


        if (len(tw2) > 3):
            if (np.abs((tw2[-1] - tw2[-2])) < 1e-4 and np.abs((self.all_error[-1] - self.all_error[-2])/self.all_error[-2])<5e-4):
                if (self.dataform[k].lrate < 1e2):
                    self.dataform[k].lrate = self.dataform[k].lrate * 1.5
                    print("speeding up")
        self.cur_par_count += 1
        newargs[k] = a
        if(self.random_search):
            if (len(allerr) > 10):
                avg10b = np.average(allerr[-10:-5])
                avg5b = np.average(allerr[-5:-1])
                stdev5b = np.std(allerr[-5:-1])
                max5b = max(allerr[-5:-1])
                if ( avg5b > 10 and stdev5b < max5b /20  and done==False):
                    if (self.reset_count < self.resetmax):
                        self.reset_count += 1
                    else:
                        self.reset_count = 0
                        if ((avg10b + avg5b) < min(allerr) * 2):
                            newargs = self.startNewPar(curpar=k, center=False, curParam=False)
                            print("newargs a avg10b=" + str(avg10b) + " avg5b=" + str(avg5b) + " min:" + str(min(allerr)) +" "+str(self.iteration))
                        else:
                            if(close==False):
                                newargs = self.startNewPar(curpar=k, center=True, curParam=False)
                                print("newargs b "+str(self.iteration))

        return newargs
    def run_finished(self):
        if(self.manual_mode!=True):
            self.showBest(None)
        self.figret.pyplt.close('all')
        print("Finished")
        self.complete = True

    def isErrorBetterThanOtherError(self):
        self.errorBetterCount+=1
        numofpar = len(self.args)
        lserdiff = np.abs(self.errors[self.cur_par][-1] - self.errors[self.cur_par][-2])
        if(len(self.errors)==len(self.args)):
            ret = True
            for i in range(numofpar):
                if(i == self.cur_par):
                    continue
                terdiff = np.abs(self.errors[i][-1] - self.errors[i][-2])
                if(terdiff<lserdiff):
                    ret = False
                    break
        else:
            ret = False
        if(ret==True and self.errorBetterCount>self.errorBetterCountMax):
            self.errorBetterCount=0
            ret = False
        return ret

    def startNewPar(self,curpar, center=False,curParam=True,completelyrandom=False):
        args = self.args
        newargs = []
        for i in range(len(args)):
            amin = self.dataform[i].min
            amax = self.dataform[i].max
            uselog = False
            top = amax-amin
            top = top+1
            bottom = 1
            if(np.abs(top/bottom)>100):
                uselog =True
            if(curParam==True):
                if(completelyrandom == False):
                    if i == curpar:

                        a = args[self.cur_par]
                        minerind=np.argmin(self.all_error)
                        matr = np.matrix(self.oldargs)
                        bestarg = matr[minerind,i]

                        mu, sigma = bestarg, np.abs(amax-amin)/10  # mean and standard deviation

                        if (center and min(self.all_error)>self.close_error):
                            sigma=np.abs(amax-amin)/100
                            mu = (np.abs(amax - amin)/2)+ amin
                        elif(min(self.all_error)<self.close_error):
                            sigma=np.abs(amax-amin)/10000
                        s = np.random.normal(mu, sigma, 1)
                        if(s<amin):
                            s=amin
                        elif(s>amax):
                            s=amax
                        newargs.append(float(s))
                    else:
                        newargs.append(float(args[i]))
                else:
                    if i == curpar:
                        if (uselog):
                            r = np.random.uniform(np.log10(amin), np.log10(amax))
                            newargs.append(float(10 ** r))
                        else:
                            r = np.random.uniform(amin, amax)
                            newargs.append(float(r))
                    else:
                        newargs.append(float(args[i]))

            else:
                if(completelyrandom==False):
                    if(len(self.all_error)>10):
                        a = args[i]
                        minerind = np.argmin(self.all_error)
                        matr = np.matrix(self.oldargs)
                        bestarg = matr[minerind, i]
                        mu, sigma = bestarg, np.sqrt(amax - amin) / 10  # mean and standard deviation
                    else:
                        center=True


                    if (center):
                        sigma =np.sqrt(amax - amin) / 10
                        mu = (np.abs(amax - amin) / 2) + amin
                    s = np.random.normal(mu, sigma, 1)
                    if (s < amin):
                        s = amin
                    elif (s > amax):
                        s = amax
                    newargs.append(float(s))
                else:
                    if(uselog):
                        r = np.random.uniform(np.log10(amin), np.log10(amax))
                        newargs.append(float(10**r))
                    else:
                        r = np.random.uniform(amin,amax)
                        newargs.append(float(r))

        return tuple(newargs)

    def argMaxMinCheck(self,v):
        amin = self.dataform[self.cur_par].min
        amax = self.dataform[self.cur_par].max
        # if(type(v) is tuple):
        #     print("asdf")
        # if (v < amin):
        #     print("asdf")
        # elif (v > amax):
        #     print("asdf")

    # def checkErrorAndCount(self):
    #     if(self.)


    def errorCalc(self, newds: dataset):
        y = newds.y
        x = newds.x
        cy = list(self.ds.y)
        cx = list(self.ds.x)
        xer = None
        yer = None
        if(self.ds.x_error is not None):
            xer = list(self.ds.x_error)
        if (self.ds.y_error is not None):
            yer = list(self.ds.y_error)
        yobs=[]
        xobs = []
        for tt in range(len(cy)):
            tval,tind = find_nearest(x,cx[tt])
            yobs.append(y[tind])
            xobs.append(tval)
        chi2 = self.chi2(yobs=yobs,yexp=cy,xobs=xobs,xexp=cx,x_error=xer,y_error=yer)
        error = chi2
        self.chi = chi2

        if len(self.figret.plots)<2:
            plotty = self.figret.ax.scatter(xobs, yobs, c="C2")
            self.figret.plots.append(plotty)
        else:
            self.figret.plots[1].set_offsets(np.c_[xobs,yobs])
        self.figret.fig.canvas.draw()
        if(math.isnan(error)):
            print("Error is NAN")


        return error
    def chi2(self,yobs,yexp,xobs=None,xexp=None,y_error=None,x_error=None):
        r=0
        islog=False
        if(np.std(yobs)>100 or np.std(xobs)>100):
            yobs = list(np.log10(np.array(yobs)))
            yexp = list(np.log10(np.array(yexp)))
            xobs = list(np.log10(np.array(xobs)))
            xexp = list(np.log10(np.array(xexp)))
            islog=True
        if(xobs is None or xexp is None):
            for i in range(len(yobs)):
                if(y_error!=None):
                    if(y_error[i]>0):
                        if(islog):
                            r += ((yobs[i] - yexp[i])**2) / np.log10(y_error[i])
                        else:
                            r += ((yobs[i] - yexp[i])**2) /y_error[i]
                else:
                    r+=((yobs[i]-yexp[i])**2)

        else:
            for i in range(len(yobs)):

                if (y_error != None):
                    if(len(y_error)==2):
                        tyer = y_error[0][i] + y_error[1][i] - 2*yexp[i]
                        if(islog):
                            tyer = np.log10(tyer)
                        if (tyer > 0):
                            r += ((yobs[i] - yexp[i])**2) /(1 +tyer)
                        else:
                            r += ((yobs[i] - yexp[i]) ** 2)
                    else:
                        if (y_error[i] > 0):
                            if(islog):
                                r += ((yobs[i] - yexp[i])**2)/ (1 + np.log10(y_error[i]))
                            else:
                                r += ((yobs[i] - yexp[i])**2) /(1 + y_error[i])
                else:
                    r += ((yobs[i] - yexp[i]) ** 2)

                if (x_error != None):
                    if(len(x_error)==2):
                        txer =x_error[0][i] + x_error[1][i] - 2*xexp[i]
                        if(islog):
                            txer=np.log10(txer)
                        if (txer > 0):
                            r += ((xobs[i] - xexp[i])**2)/ (1+txer)
                        else:
                            r += ((xobs[i] - xexp[i]) ** 2)
                    else:
                        if (x_error[i] > 0):
                            if(islog):
                                r += ((xobs[i] - xexp[i])**2) / (1 + np.log10(x_error[i]))
                            else:
                                r += ((xobs[i] - xexp[i])**2)/ (1+x_error[i])
                else:
                    r += ((xobs[i] - xexp[i]) ** 2)
        return r
    def plotError(self,iter):
        if(self.errorplot==None):
            fig, ax = plt.subplots()
            plots=[]
            xs = self.oldargs_byparameter[self.cur_par]
            ys = self.errors[self.cur_par]
            newxs = []
            newys = []
            for i in range(len(ys)):
                if (ys[i] <= 10):
                    newys.append(ys[i])
                    newxs.append(xs[i])
            ax.set_ylim(-2, 11)
            plot, = ax.plot(newxs, newys,'--bo')
            plots.append(plot)
            fig.set_tight_layout(True)

            figr = figret(fig=fig,ax=ax,pyplt=plt,plots=plots)
            axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
            self.errorSlider = Slider(axframe, 'Frame', 0, len(self.args), valinit=0,valfmt='%d')
            self.errorSlider.on_changed(self.updateErrorPlot)
            fig.show()
            self.errorplot=figr
        else:
            # for i in range(len(self.errorplot.plots)):
            plot = self.errorplot.plots[0]
            # plot.set_offsets(np.c_[self.oldargs_byparameter[i],self.errors[i]])
            xs = self.oldargs_byparameter[self.cur_err_par]
            ys = self.errors[self.cur_err_par]
            newxs = []
            newys = []
            for i in range(len(ys)):
                if (ys[i] <= 10):
                    newys.append(ys[i])
                    newxs.append(xs[i])
            plot.set_xdata(newxs)
            plot.set_ydata(newys)
            ax = self.errorplot.ax
            if(len(newxs)>1):
                newxs = np.array(newxs)
                newys = np.array(newys)
                u =newxs[1:] - newxs[:-1]
                v = newys[1:] - newys[:-1]

                # self.errorplot.ax.quiver( newxs[:-1] , newys[:-1], u, v, scale_units='xy', angles='xy', scale=1.5,color='b',width=0.005,linestyle="--")

            ax.relim()
            ax.autoscale_view()
            self.errorplot.pyplt.draw()
            # plot = self.errorplot.plots[-1]
            # plot.set_offsets(np.c_[range(iter),self.all_error])

    def updateErrorPlot(self,val):
        frame = int(np.floor(self.errorSlider.val))
        self.cur_err_par=frame
        ln = self.errorplot.plots[0]
        xs = self.oldargs_byparameter[frame]
        ys = self.errors[frame]
        newxs=[]
        newys=[]
        for i in range(len(ys)):
            if(ys[i]<=10):
                newys.append(ys[i])
                newxs.append(xs[i])
        ln.set_xdata(newxs)
        ln.set_ydata(newys)
        if (len(newxs) > 1):
            newxs = np.array(newxs)
            newys = np.array(newys)
            u = newxs[1:] - newxs[:-1]
            v = newys[1:] - newys[:-1]
            #
            # self.errorplot.ax.quiver(newxs[:-1], newys[:-1], u, v, scale_units='xy', angles='xy', scale=1.5, color='b',
            #                          width=0.005, linestyle="--")

        ax = self.errorplot.ax
        # for k in range(len(xs)):
        #     label = "{:.2f}".format(k)
        #
        #     ax.annotate(label,  # this is the text
        #                 (xs[k], ys[k]),  # this is the point to label
        #                 textcoords="offset points",  # how to position the text
        #                 xytext=(0, 10),  # distance from text to points (x,y)
        #                 ha='center')

        ax.set_title(frame)
        ax.relim()
        ax.autoscale_view()
        self.errorplot.pyplt.draw()