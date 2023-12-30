#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
import dataclasses
import datetime, time, os.path as p
import matplotlib.pyplot as plt
from typing import Any, Callable
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import erf
from cic.models2 import cosmology, Cosmology
from cic.models2.cosmology import FlatLambdaCDM
from cic.models2.hod import HaloModel

# save the doculment as a pdf
def print_report(document: list, path: str, metadata: dict = None):
    metadata = metadata or {}
    with PdfPages(path, metadata = metadata) as pdf:
        for page in document: pdf.savefig(page)
    return

# estimate execution time
def estimate_exectime(func, *args, **kwargs):
    __timestamp = time.time_ns()
    res = func(*args, **kwargs) 
    __timestamp = time.time_ns() - __timestamp
    return res, [func.__qualname__, __timestamp / 1000]

# return the current time as a datetime object
def localtime() -> datetime.datetime:
    t = time.localtime()
    return datetime.datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

# generate a comparison repport for two functions
def comparison_test(f1: Callable, # functions to compare: should have similar call signature
                    f2: Callable,
                    x: Any,       # first argument to the function (used to compare)
                    args: tuple = (),          # other arguments 
                    title: str | None = None,  # test report page title
                    text: str | None = None,   # extra text
                    xlabel: str | None = None, # axis labels
                    ylabel: str | None = None, 
                    label1: str | None = None, # data labels
                    label2: str | None = None,
                    plot_scale: str | None = None, # plot scale: loglog, semilogx or semilogy
                    error_limit: float | None = None, # error axis upper limit 
                    _scale: float = 1.0, ) -> Any:
    ##################################################################################################
    #                                 step 1: comparing functions                                    #           
    ##################################################################################################
    y1, y2 = f1(x, *args), f2(x, *args)
    relerr = np.abs(y2 - y1) / np.where( y1 == 0, 1., np.abs(y1) ) # relative error
    arg_maxerr = np.argmax(relerr)
    maxerr     = relerr[arg_maxerr]
    x_maxerr   = x[arg_maxerr] 
    ##################################################################################################
    #                                 step 2: generating report                                      #       
    ##################################################################################################
    page = plt.figure(figsize = (_scale*8.3, _scale*11.7)) # a4 size 
    # title
    ax = page.add_axes([0.12, 0.9, 0.8, 0.02])
    ax.axis('off')
    ax.set_title(title, fontdict = {'fontsize': 14, 'fontweight': 'bold',})
    ax.text(0.5,  0.0, "%s $vs.$ %s" % (f1.__qualname__, f2.__qualname__), fontsize = 10, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes) 
    ax.text(0.5, -2.0, text, fontsize = 10, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes) 
    # comparison plot
    ax = page.add_axes([0.12, 0.4, 0.8, 0.35])
    ax.set_ylabel(ylabel)
    if   plot_scale == 'loglog'  : ax.loglog()
    elif plot_scale == 'semilogx': ax.semilogx()
    elif plot_scale == 'semilogy': ax.semilogy()
    ax.plot(x, y1, '-', color = '#0055d4', label = label1 or '%s' % f1.__qualname__)
    ax.plot(x, y2, 'o', color = '#0055d4', label = label2 or '%s' % f2.__qualname__)
    ax.legend(ncols = 2, bbox_to_anchor = (0, 1, 1, 1), loc = 'lower left', mode = 'expand', alignment = 'left', frameon = False)
    # rel. error
    ax = page.add_axes([0.12, 0.3, 0.8, 0.10], sharex = ax)
    if plot_scale in ( 'semilogx', 'loglog' ): ax.semilogx()
    ax.plot(x, relerr, 'o-', color = '#0055d4')
    ax.set(ylim = (-1., error_limit or np.max(relerr) + 1), xlabel = xlabel, ylabel = '$rel. error$')
    #
    ax = page.add_axes([0.12, 0.2, 0.8, 0.05], sharex = ax)
    ax.axis('off')
    ax.text(0.0, 0.5, "$Max.~relative~error:$ %.3g at %s=%g" % (maxerr, xlabel or 'x', x_maxerr), fontsize = 10, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    #
    return page

# test halomodel
class TestHaloModel(HaloModel):
    def __init__(self, 
                 cosmology: object, 
                 overdensity: float = 200.,
                 mmin: float = 1e+08, 
                 msat: float = 1e+12,
                 mscale: float = 0.2, 
                 alpha: float = 1., ) -> None:
        super().__init__(cosmology, overdensity)
        # model parameters
        self.mmin   = mmin
        self.msat   = msat
        self.mscale = mscale
        self.alpha  = alpha
        self.mcut   = np.sqrt(mmin)**-1 #np.exp( 1.18*np.log(mmin) - 1.28 )
        return

    def centralCount(self, m: float) -> float:
        x = ( np.log(m) - np.log(self.mmin) ) / ( np.sqrt(2) * self.mscale )
        return 0.5 * ( 1. + erf(x) )

    def satelliteFraction(self, m: float) -> float:
        return ( ( np.subtract(m, self.mcut) ) / self.msat )**self.alpha  

class GeneralTest:
    def __init__(self) -> None:
        # object to store named values with properties
        @dataclasses.dataclass
        class Value:
            # name or description of the quantity
            name: str
            # symbol for the quantity
            symbol: str
            # value of the quantity
            value: float
            # extra information
            extra: str
            # return the value as str of the for {name},{symbol}={value}
            @property
            def string(self) -> str: return f'{self.name}, {self.symbol}\n{self.value:.2f}'
            # return as a list of sts
            @property
            def list(self) -> list: return [ self.name, self.symbol, f'{self.value:.3f}', self.extra ]
        # test cosmology model: plank 2018
        model = cosmology('plank18')
        model.link(power_spectrum = 'eisenstein98_zb', 
                   window         = 'tophat', 
                   mass_function  = 'tinker08', 
                   halo_bias      = 'tinker10', 
                   cmreln         = 'zheng07',
                   halo_profile   = 'nfw',   )
        self.model   = model
        # test halo model
        self.overdensity = 200
        self.halo_model = TestHaloModel(model, self.overdensity)
        # component density data
        self.densities = [Value('Baryons'    , '$\\Omega_b$'   , model.Ob0                         , '#0055d4' ), 
                          Value('Neutrino'   , '$\\Omega_\\nu$', model.Onu0                        , '#0066ff' ), 
                          Value('Dark-matter', '$\\Omega_{dm}$', model.Om0 - model.Ob0 - model.Onu0, '#003380' ), 
                          Value('Dark-energy', '$\\Omega_{de}$', model.Ode0                        , '#c87137' ),]
        # other parameters table
        self.other_params = [Value( 'Hubble parameter'          , '$H_0$'      , 100*model.h  , 'km/s/Mpc' ), 
                             Value( 'Power spectrum index'      , '$n_s$'      , model.ns     , ''         ),
                             Value( 'Variance at 8 Mpc/h'       , '$\\sigma_8$', model.sigma8 , ''         ),
                             Value( 'CMB temperature'           ,'$T_{cmb}$'   , model.Tcmb0  , 'K'        ),
                             Value( 'No. Neutrino species'      ,'$N_\\nu$'    , model.Nnu    , ''         ),
                             Value( 'Curvature density'         ,'$\\Omega_k$' , model.Ok0    , ''         ),
                             Value( 'Dark energy eqn. of state' ,'$w_0$'       , model.w0     , ''         ),
                             Value( 'Dark energy eqn. of state' ,'$w_a$'       , model.wa     , ''         ),]
        # execution time
        self.exec_time = []
        self._rows_per_page = 25
        self._first_page = True
        self._first_row = 0

    def newFigure(self): 
        return plt.figure(figsize = (8.3, 11.7)) # a4 size

    def generate_test_report(self) -> list:
        __time = time.time()
        print("Info: starting test. time:", time.asctime())
        document = []
        for __name, __value in self.__class__.__dict__.items():
            if not __name.startswith('figure_'): continue
            print(f"Info: running '{__name}'...")
            fig = __value(self)
            if fig is None: continue
            document.append(fig)
        print("Info: running 'execution_time_table'...")
        while 1:
            fig = self.execution_time_table()
            if fig is not None:
                document.append(fig)
                continue
            break
        __time = time.time() - __time
        print(f"Test completed in {__time:.3f} second. time:", time.asctime())
        return document
    
    def execution_time_table(self):
        if len(self.exec_time) == 0 or self._first_row >= len(self.exec_time): return 
        start = self._first_row
        stop  = min(len(self.exec_time), start + self._rows_per_page)
        table = self.exec_time[start:stop]
        fig = self.newFigure()
        # creating the figure
        # title
        ax = fig.add_axes([0.12, 0.85, 0.8, 0.05])
        ax.axis('off')
        ax.set_title("Execution time" + ('' if self._first_page else ' (contd.)'), fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        table = ax.table(colLabels = [ 'Function', 'Input total shape', 'Time (us)'], 
                         cellText  = table,
                         rowLabels = [ f'{i+1:6d}' for i in range(start, stop) ],
                         colWidths = [ 0.6, 0.25, 0.15],
                         colLoc = 'right',
                         cellLoc = 'right')
        for (row, col), cell in table.get_celld().items():
            if not row or col == -1:
                cell.set_text_props(fontproperties = {'weight': 'bold'})
            if not row:
                cell.set_text_props(fontproperties = {'weight': 'bold'}, color = 'white')
                cell.set(color = '#003380')
            cell.set(color = '#003380' if not row else '#fff' if row%2 else '#d5e5ff')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        self._first_row = stop
        self._first_page = False
        return fig
    
    ######################################################################################
    # cosmology model test plots
    ######################################################################################
    
    def figure_parameterTable(self):
        # creating the figure
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.1, 0.9, 0.8, 0.01])
        ax.axis('off')
        ax.set_title("Cosmology model details", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        # density pie chart
        ax = fig.add_axes([0.1, 0.38, 0.8, 0.5])
        ax.pie(list( map(lambda __x: __x.value , filter(lambda __x: __x.value != 0, self.densities)) ), 
               labels = list( map(lambda __x: __x.string, filter(lambda __x: __x.value != 0, self.densities)) ), 
               colors = list( map(lambda __x: __x.extra , filter(lambda __x: __x.value != 0, self.densities)) ), 
               textprops = { 'fontsize': 10 }, 
               wedgeprops = { 'width': 0.5 }, 
               startangle = 45, )
        ax.set_title("Universe Composition", fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # other parameters table
        ax = fig.add_axes([0.1, 0.35, 0.8, 0.0001])
        ax.axis('off')
        ax.set_title("Other parameters", fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        table = ax.table(colLabels = [ 'Parameter', 'Symbol', 'Value', 'Unit'], 
                         cellText  = list( map(lambda __x: __x.list, self.other_params) ),
                         colWidths = [ 0.5, 0.15, 0.15, 0.2],
                         cellLoc = 'right')
        for (row, col), cell in table.get_celld().items():
            if not row:
                cell.set_text_props(fontproperties = {'weight': 'bold'}, color = 'white')
                cell.set(color = '#003380')
            cell.set(color = '#003380' if not row else '#fff' if row%2 else '#d5e5ff')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        return fig
    
    def figure_densityFunctions(self):
        z = np.linspace(0, 5, 11)
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.12, 0.9, 0.81, 0.01])
        ax.axis('off')
        ax.set_title("Densities", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        ax.text(0.5, 0.0, 
                 'Densities in unit of present day critical density (approx. $2.8\\times10^{11}$ $h^2~Msun/Mpc^3$)', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # density evolutions
        ax = fig.add_axes([0.12, 0.65, 0.78, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.model.Om, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y, 'o-', color = '#0066ff', label = 'Matter' )
        y, [__fname, __texec] = estimate_exectime(self.model.Ode, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y, 'o-', color = '#c87137', label = 'Dark-energy' )
        ax.set(xlabel = 'z', ylabel = '$\\Omega_x(z)$') 
        ax.set_title('Matter and DE', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        ax.legend()
        # critical density
        ax = fig.add_axes([0.12, 0.38, 0.78, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.model.E, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y**2, 'o-', color = '#0066ff' )
        ax.set(xlabel = 'z', ylabel = '$E^2(z)$') 
        ax.set_title('Critical Density $\\propto E^2(z)$', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        return fig
    
    def figure_distances(self):
        z = np.linspace(0, 5, 11)
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.12, 0.93, 0.81, 0.01])
        ax.axis('off')
        ax.set_title("Distances", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        ax.text(0.5, 0.0, 
                 'Distances are in unit of Mpc/h', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        #
        ax = fig.add_axes([0.12, 0.65, 0.81, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.model.comovingDistance, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y , 'o-', color = '#0066ff' )
        ax.set(xlabel = 'z', ylabel = '$x(z)$') 
        ax.set_title('Comoving distance', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        #
        ax = fig.add_axes([0.12, 0.38, 0.35, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.model.angularDiameterDistance, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y , 'o-', color = '#0066ff' )
        ax.set(xlabel = 'z', ylabel = '$d_A(z)$') 
        ax.set_title('Angular diameter distance', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        #
        ax = fig.add_axes([0.58, 0.38, 0.35, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.model.luminocityDistance, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y, 'o-', color = '#0066ff' )
        ax.set(xlabel = 'z', ylabel = '$d_L(z)$') 
        ax.set_title('Luminocity distance', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # angular size table
        ax = fig.add_axes([0.2, 0.25, 0.7, 0.05])
        ax.axis('off')
        ax.text(0.45, 0.5, 
                 'Redshift z in column labels, angular size in arcsec as row labels. Physical size in Mpc/h', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("Angulr size to Physical size conversion at different z", 
                     fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        z     = [1., 2., 3., 4., 5., 10.]
        angle = [100., 200., 500., 1000.]
        table = ax.table(colLabels = [ f' {zi:6g} ' for zi in z ], 
                         rowLabels = [ f' {t:>6g}" ' for t in angle ],
                         cellText  = [ [ '{:.3f}'.format( self.model.angularSize(zi, t, inverse = True) ) for zi in z ] for t in angle ],
                         cellLoc = 'right')
        for (row, col), cell in table.get_celld().items():
            if not row or col == -1:
                cell.set_text_props(fontproperties = {'weight': 'bold'})
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        return fig
    
    def figure_time(self):
        z = np.linspace(0, 5, 11)
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.12, 0.93, 0.81, 0.01])
        ax.axis('off')
        ax.set_title("Time", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        ax.text(0.5, 0.0, 
                 'Current age of universe: {:.3f} Gyr'.format( self.model.age() ), 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        #
        ax = fig.add_axes([0.12, 0.65, 0.68, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.model.age, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y, 'o-', color = '#0066ff', label = 'Age' )
        y, [__fname, __texec] = estimate_exectime(self.model.hubbleTime, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y, 'o-', color = '#c87137', label = 'Hubble time' )
        ax.set(xlabel = 'z', ylabel = '$t(z)$') 
        ax.set_title('Age of the universe (Gyr)', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        ax.legend(bbox_to_anchor = (1, 0, 1, 1), loc = 'center left', frameon = False)
        return fig
    
    def figure_structure(self):
        z = np.array([0., 1., 2.])
        x = np.logspace(-3, 3, 11)
        colors = ['#003380', '#0066ff', '#0055d4']
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.12, 0.93, 0.81, 0.01])
        ax.axis('off')
        ax.set_title("Density statistics", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        ax.text(0.5, 0.0, 
                 f'Using the power spectrum model "{self.model.power_spectrum.__class__.__name__}" with a "{self.model.window_function.__class__.__name__}" filter', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax = fig.add_axes([0.12, 0.82, 0.8, 0.001])
        for __z, __col in zip(z, colors):
            ax.plot([], [], 'o-', color = __col, label = '%g' % __z )
        ax.axis('off')
        ax.legend(title = 'Redshift, z', ncols = len(z), bbox_to_anchor = (0, 1, 1, 1), loc = 'lower left', mode = 'expand', alignment = 'left', frameon = False)
        # power spectrum
        ax = fig.add_axes([0.12, 0.60, 0.35, 0.2])
        __texec_max = 0
        for __z, __col in zip(z, colors):
            y, [__fname, __texec] = estimate_exectime(self.model.matterPowerSpectrum, x, __z) 
            __texec_max = max(__texec_max, __texec)
            ax.loglog( x, y, 'o-', color = __col, label = '%g' % __z )
        self.exec_time.append([__fname, str(np.shape(x) + (1,)), '%.1f' % __texec_max])
        ax.set(xlabel = 'k', ylabel = '$P(k, z)$') 
        ax.set_title('Power spectrum', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # correlation
        ax = fig.add_axes([0.58, 0.60, 0.35, 0.2])
        __texec_max = 0
        for __z, __col in zip(z, colors):
            y, [__fname, __texec] = estimate_exectime(self.model.matterCorrelation, x, __z) 
            __texec_max = max(__texec_max, __texec)
            ax.loglog( x, y, 'o-', color = __col, label = '%g' % __z )
        self.exec_time.append([__fname, str(np.shape(x) + (1,)), '%.1f' % __texec_max])
        ax.set(xlabel = 'r', ylabel = '$\\xi(r, z)$') 
        ax.set_title('Correlation (average)', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # variance
        ax = fig.add_axes([0.12, 0.33, 0.35, 0.2])
        __texec_max = 0
        for __z, __col in zip(z, colors):
            y, [__fname, __texec] = estimate_exectime(self.model.matterVariance, x, __z) 
            __texec_max = max(__texec_max, __texec)
            ax.loglog( x, y , 'o-', color = __col, label = '%g' % __z )
        self.exec_time.append([__fname, str(np.shape(x) + (1,)), '%.1f' % __texec_max])
        ax.set(xlabel = 'r', ylabel = '$\\sigma^2(r, z)$') 
        ax.set_title('Variance', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # variance derivative
        ax = fig.add_axes([0.58, 0.33, 0.35, 0.2])
        __texec_max = 0
        for __z, __col in zip(z, colors):
            y, [__fname, __texec] = estimate_exectime(self.model.matterVariance, x, __z, 1) 
            __texec_max = max(__texec_max, __texec)
            ax.semilogx( x, y , 'o-', color = __col, label = '%g' % __z )
            break
        self.exec_time.append([__fname + '(deriv=1)', str(np.shape(x) + (1,)), '%.1f' % __texec_max])
        ax.set(xlabel = 'r', ylabel = '$\\frac{d\\ln\\sigma}{d\\ln r}(r, z)$') 
        ax.set_title('Variance 1-st log-derivative', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # ax.legend(title = 'Redshift, z')
        # growth factors
        ax = fig.add_axes([0.12, 0.11, 0.68, 0.15])
        z = np.linspace(0, 5, 11)
        y, [__fname, __texec] = estimate_exectime(self.model.dplus, z, 0) 
        self.exec_time.append([__fname + '(deriv=0)', str(np.shape(x) + (1,)), '%.1f' % __texec])
        ax.plot( z, y, 'o-', color = '#0066ff', label = '$D_+(z)$' )
        y, [__fname, __texec] = estimate_exectime(self.model.dplus, z, 1) 
        self.exec_time.append([__fname + '(deriv=1)', str(np.shape(x) + (1,)), '%.1f' % __texec])
        ax.plot( z, y, 'o-', color = '#c87137', label = '$f_+(z)$' )
        ax.set(xlabel = 'z', ylabel = '$D_+(z)$ or $f_+(z)$') 
        ax.set_title('Growth factors', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        ax.legend(bbox_to_anchor = (1, 0, 1, 1), loc = 'center left', frameon = False)
        return fig
    
    def figure_filters(self):
        from cic.models2 import WindowFunction
        x = np.linspace(-10, 10, 51)
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.12, 0.93, 0.81, 0.01])
        ax.axis('off')
        ax.set_title("Different filter functions", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        #
        ax = fig.add_axes([0.12, 0.65, 0.78, 0.2])
        for __filt, __colour in [('tophat', '#0066ff'), ('gaussian', '#c87137')]:
            y, [__fname, __texec] = estimate_exectime(WindowFunction.available.get(__filt).call, x, 0) 
            self.exec_time.append([__fname + '(deriv=0)', str(np.shape(x)), '%.1f' % __texec])
            ax.plot( x, y, 'o-', color = __colour, label = __filt )
        ax.set(xlabel = 'x', ylabel = '$w(x)$') 
        ax.legend(ncols = 2, bbox_to_anchor = (0, 1, 1, 1), loc = 'lower left', mode = 'expand', frameon = False)
        #
        ax = fig.add_axes([0.12, 0.38, 0.78, 0.2])
        for __filt, __colour in [('tophat', '#0066ff'), ('gaussian', '#c87137')]:
            y, [__fname, __texec] = estimate_exectime(WindowFunction.available.get(__filt).call, x, 1) 
            self.exec_time.append([__fname + '(deriv=1)', str(np.shape(x)), '%.1f' % __texec])
            ax.plot( x, y, 'o-', color = __colour, label = __filt )
        ax.set(xlabel = 'x', ylabel = '$w\'(x)$') 
        ax.set_title('1-st derivative', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        #
        ax = fig.add_axes([0.12, 0.11, 0.78, 0.2])
        for __filt, __colour in [('tophat', '#0066ff'), ('gaussian', '#c87137')]:
            y, [__fname, __texec] = estimate_exectime(WindowFunction.available.get(__filt).call, x, 2) 
            self.exec_time.append([__fname + '(deriv=2)', str(np.shape(x)), '%.1f' % __texec])
            ax.plot( x, y, 'o-', color = __colour, label = __filt )
        ax.set(xlabel = 'x', ylabel = '$w\'\'(x)$') 
        ax.set_title('2-nd derivative', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        return fig
    
    def figure_halostats(self):
        z = np.array([0., 1., 2.])
        m = np.logspace(6, 17, 11)
        colors = ['#003380', '#0066ff', '#0055d4']
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.12, 0.93, 0.81, 0.01])
        ax.axis('off')
        ax.set_title("Halo statistics", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        ax.text(0.5, -2.0, 
                 '\n'.join([f'Using the mass-function model "{self.model.mass_function.__class__.__name__}", bias model "{self.model.halo_bias.__class__.__name__}", and concentration-mass relation',
                            f'"{self.model.halo_cmreln.__class__.__name__}", with power spectrum model "{self.model.power_spectrum.__class__.__name__}" with a ', 
                            f'"{self.model.window_function.__class__.__name__}" filter. Masses are in Msun/h unit and overdensity $\\Delta_m$ = 200.', ]), 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax = fig.add_axes([0.12, 0.82, 0.78, 0.001])
        for __z, __col in zip(z, colors):
            ax.plot([], [], 'o-', color = __col, label = '%g' % __z )
        ax.axis('off')
        ax.legend(title = 'Redshift, z', ncols = len(z), bbox_to_anchor = (0, 1, 1, 1), loc = 'lower left', mode = 'expand', alignment = 'left', frameon = False)
        # mass function
        __texec_max = 0
        ax = fig.add_axes([0.12, 0.60, 0.78, 0.2])
        for __z, __col in zip(z, colors):
            y, [__fname, __texec] = estimate_exectime(self.model.haloMassFunction, m, __z, self.overdensity) 
            __texec_max = max(__texec_max, __texec)
            ax.loglog( m, y, 'o-', color = __col, label = '%g' % __z )
        self.exec_time.append([__fname, str(np.shape(m) + (1, 1,)), '%.1f' % __texec_max])
        ax.set(xlabel = 'm', ylabel = '$\\frac{dn}{d\\ln m}(m, z)$') 
        ax.set_title('Mass function', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # bias
        ax = fig.add_axes([0.12, 0.33, 0.78, 0.2])
        __texec_max = 0
        for __z, __col in zip(z, colors):
            y, [__fname, __texec] = estimate_exectime(self.model.haloBias, m, __z, self.overdensity) 
            __texec_max = max(__texec_max, __texec)
            ax.loglog( m, y, 'o-', color = __col, label = '%g' % __z )
        self.exec_time.append([__fname, str(np.shape(m) + (1, 1,)), '%.1f' % __texec_max])
        ax.set(xlabel = 'm', ylabel = '$b_h(m, z)$') 
        ax.set_title('Bias', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # concentration
        __texec_max = 0
        ax = fig.add_axes([0.12, 0.06, 0.78, 0.2])
        for __z, __col in zip(z, colors):
            y, [__fname, __texec] = estimate_exectime(self.model.haloConcentration, m, __z, self.overdensity) 
            __texec_max = max(__texec_max, __texec)
            ax.loglog( m, y, 'o-', color = __col, label = '%g' % __z )
        self.exec_time.append([__fname, str(np.shape(m) + (1, 1,)), '%.1f' % __texec_max])
        ax.set(xlabel = 'm', ylabel = '$c(m, z)$') 
        ax.set_title('Concentration', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        return fig
    
    def figure_halostats2(self):
        z = np.array([0., 1., 2.])
        x = np.logspace(-4, 2, 11)
        colors = ['#003380', '#0066ff', '#0055d4']
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.12, 0.93, 0.81, 0.01])
        ax.axis('off')
        ax.set_title("Halo statistics", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        ax.text(0.5, -2.0, 
                 '\n'.join([f'Halo profile "{self.model.halo_profile.__class__.__name__}" with concentration-mass relation "{self.model.halo_cmreln.__class__.__name__}".',
                            'Masses are in Msun/h, distance in Mpc/h (wavenumber k in h/Mpc). \nReal space profile is truncated at virial radius.']), 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax = fig.add_axes([0.12, 0.82, 0.8, 0.001])
        for __z, __col in zip(z, colors):
            ax.plot([], [], 'o-', color = __col, label = '%g' % __z )
        ax.axis('off')
        ax.legend(title = 'Redshift, z', ncols = len(z), bbox_to_anchor = (0, 1, 1, 1), loc = 'lower left', mode = 'expand', alignment = 'left', frameon = False)
        # real density
        __ypos, __first = 0.6, True
        __texec_max = 0
        for m in ( 1e+08, 1e+12, 1e+16 ):
            ax = fig.add_axes([0.12, __ypos, 0.35, 0.2])
            for __z, __col in zip(z, colors):
                y, [__fname, __texec] = estimate_exectime(self.model.haloProfile, x, m, __z, self.overdensity, False, True) 
                __texec_max = max(__texec_max, __texec)
                ax.loglog( x, y*1e-07 , 'o-', color = __col, label = '%g' % __z )
            ax.set(xlabel = 'r', ylabel = '$\\rho(r; m, z) \\times 10^{-7}$') 
            ax.set_title(('Real profile \n' if __first else '') + 'm = %g'%m, fontdict = {'fontsize': 10, 'fontweight': 'bold',})
            __ypos -= 0.275
            __first = False
        self.exec_time.append([__fname + '(fourier_transform=0)', str(np.shape(x) + (1, 1)), '%.1f' % __texec_max])
        # fourier profile
        __ypos, __first = 0.6, True
        __texec_max = 0
        for m in ( 1e+08, 1e+12, 1e+16 ):
            ax = fig.add_axes([0.58, __ypos, 0.35, 0.2])
            for __z, __col in zip(z, colors):
                y, [__fname, __texec] = estimate_exectime(self.model.haloProfile, x, m, __z, self.overdensity, True, False) 
                __texec_max = max(__texec_max, __texec)
                ax.semilogx( x, y, 'o-', color = __col, label = '%g' % __z )
            ax.set(xlabel = 'k', ylabel = '$u(k; m, z)$') 
            ax.set_title(('Fourier profile \n' if __first else '') + 'm = %g'%m, fontdict = {'fontsize': 10, 'fontweight': 'bold',})
            __ypos -= 0.275
            __first = False
        self.exec_time.append([__fname + '(fourier_transform=1)', str(np.shape(x) + (1, 1)), '%.1f' % __texec_max])
        return fig
    
    ######################################################################################
    # halo model test plots
    ######################################################################################

    def figure_hod_model(self):
        m = np.logspace(7, 12, 11)
        # creating the figure
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.1, 0.9, 0.8, 0.01])
        ax.axis('off')
        ax.set_title("Halo model details", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        # parameters table
        ax = fig.add_axes([0.1, 0.85, 0.8, 0.0001])
        ax.axis('off')
        ax.set_title("Halo model parameters", fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        hm = self.halo_model
        table = ax.table(colLabels = [ '$M_{min}$', '$M_{sat}$', '$\\sigma$', '$M_{cut}$', '$\\alpha$' ], 
                         cellText  = [list( map(lambda __x: '%.2g'%__x, [ hm.mmin, hm.msat, hm.mscale, hm.mcut, hm.alpha]) )],
                         cellLoc = 'right')
        for (row, col), cell in table.get_celld().items():
            if not row:
                cell.set_text_props(fontproperties = {'weight': 'bold'}, color = 'white')
                cell.set(color = '#003380')
            cell.set(color = '#003380' if not row else '#d5e5ff' if row%2 else '#fff')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        # central count eqn.
        ax = fig.add_axes([0.1, 0.72, 0.8, 0.05])
        ax.axis('off')
        ax.text(0.48, 0.5, 'Central galaxy count:', fontdict = {'fontsize': 10, 'fontweight': 'bold'}, 
                horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, )
        ax.text(0.5, 0.5, 
                '$N_c(m) = \\frac{1}{2} \\left[ 1 + erf \\left( \\frac{\\ln m - \\ln M_{min}}{\\sqrt{2}\\sigma} \\right) \\right]$', 
                fontdict = {'fontsize': 12}, 
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, )
        # central count
        ax = fig.add_axes([0.1, 0.57, 0.8, 0.15])
        y1, [__fname, __texec] = estimate_exectime(self.halo_model.centralCount, m) 
        self.exec_time.append([__fname, str(np.shape(m)), '%.1f' % __texec])
        ax.semilogx(m, y1, 'o-', color = '#0066ff')
        ax.set(xlabel = 'm', ylabel = '$N_c(m)$')
        # satellite count eqn.
        ax = fig.add_axes([0.1, 0.48, 0.8, 0.05])
        ax.axis('off')
        ax.text(0.48, 0.5, 'Satellite galaxy count:', fontdict = {'fontsize': 10, 'fontweight': 'bold'}, 
                horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, )
        ax.text(0.52, 0.5, 
                '$N_s(m) = N_c(m) \\left( \\frac{m - M_{cut}}{M_{sat}} \\right)^{\\alpha}$', 
                fontdict = {'fontsize': 12}, 
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, )
        # satellite count
        ax = fig.add_axes([0.1, 0.33, 0.8, 0.15])
        y, [__fname, __texec] = estimate_exectime(self.halo_model.satelliteFraction, m) 
        self.exec_time.append([__fname, str(np.shape(m)), '%.1f' % __texec])
        ax.semilogx(m, y1*y, 'o-', color = '#0066ff')
        ax.set(xlabel = 'm', ylabel = '$N_s(m)$')
        # total count
        ax = fig.add_axes([0.1, 0.08, 0.8, 0.15])
        y, [__fname, __texec] = estimate_exectime(self.halo_model.totalCount, m) 
        self.exec_time.append([__fname, str(np.shape(m)), '%.1f' % __texec])
        ax.semilogx(m, y, 'o-', color = '#c87137')
        ax.set(xlabel = 'm', ylabel = '$N(m)$')
        ax.set_title('Total galaxy count', fontdict = {'fontsize': 10, 'fontweight': 'bold'})
        return fig
    
    def figure_hodstats(self):
        z = np.linspace(0, 5, 11)
        # creating the figure
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.1, 0.9, 0.8, 0.01])
        ax.axis('off')
        ax.set_title("Galaxy statistics", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        ax.text(0.5, 0.0, 
                 f"Integration details: mass range: [{self.halo_model.ma:.2g}, {self.halo_model.mb:.2g}], {self.halo_model.mpts} points.",
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # no. density
        ax = fig.add_axes([0.12, 0.65, 0.78, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.halo_model.galaxyDensity, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.semilogy( z, y, 'o-', color = '#0066ff' )
        ax.set(xlabel = 'z', ylabel = '$n_g(z)$') 
        ax.set_title('Galaxy no. density', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # average mass
        ax = fig.add_axes([0.12, 0.38, 0.78, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.halo_model.averageHaloMass, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.semilogy( z, y, 'o-', color = '#0066ff' )
        ax.set(xlabel = 'z', ylabel = '$\\widebar{m}(z)$') 
        ax.set_title('Average halo mass', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        # effective bias
        ax = fig.add_axes([0.12, 0.11, 0.81, 0.2])
        y, [__fname, __texec] = estimate_exectime(self.halo_model.effectiveBias, z) 
        self.exec_time.append([__fname, str(np.shape(z)), '%.1f' % __texec])
        ax.plot( z, y, 'o-', color = '#0066ff' )
        ax.set(xlabel = 'z', ylabel = '$b_{eff}(z)$') 
        ax.set_title('Effective galaxy bias', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
        return fig
    
    def figure_hod_power(self):
        z = np.array([0., 1., 2.])
        x = np.logspace(-4, 2, 11)
        colors = ['#003380', '#0066ff', '#0055d4']
        # creating the figure
        fig = self.newFigure()
        # title
        ax = fig.add_axes([0.1, 0.9, 0.8, 0.01])
        ax.axis('off')
        ax.set_title("Galaxy power spectrum", fontdict = {'fontsize': 14, 'fontweight': 'bold',})
        ax.text(0.5, 0.0, 
                 f"Integration details: mass range: [{self.halo_model.ma:.2g}, {self.halo_model.mb:.2g}], {self.halo_model.mpts} points.",
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        #
        ax = fig.add_axes([0.12, 0.82, 0.8, 0.001])
        for __z, __col in zip(z, colors):
            ax.plot([], [], 'o-', color = __col, label = '%g' % __z )
        ax.axis('off')
        ax.legend(title = 'Redshift, z', ncols = len(z), bbox_to_anchor = (0, 1, 1, 1), loc = 'lower left', mode = 'expand', alignment = 'left', frameon = False)
        # power spectrum
        __ypos, __first = 0.65, True
        __texec_max = 0
        for __type in ( 'cs', 'ss', '2h', '1h+2h' ):
            ax = fig.add_axes([0.12, __ypos, 0.35, 0.15])
            for __z, __col in zip(z, colors):
                y, [__fname, __texec] = estimate_exectime(self.halo_model.galaxyPowerSpectrum, x, __z, __type) 
                __texec_max = max(__texec_max, __texec)
                ax.loglog( x, y, 'o-', color = __col, label = '%g' % __z )
            ax.set(xlabel = 'k', ylabel = '$P_{%s}(k, z)$'%__type) 
            if __first:
                ax.set_title('Power spectrum', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
            __ypos -= 0.2
            __first = False
        self.exec_time.append([__fname, str(np.shape(x) + (1,)), '%.1f' % __texec])
        # correlation
        __ypos, __first = 0.65, True
        for __type in ( 'cs', 'ss', '2h', '1h+2h' ):
            ax = fig.add_axes([0.58, __ypos, 0.35, 0.15])
            for __z, __col in zip(z, colors):
                y, [__fname, __texec] = estimate_exectime(self.halo_model.galaxyCorrelation, x, __z, __type) 
                __texec_max = max(__texec_max, __texec)
                ax.loglog( x, y, 'o-', color = __col, label = '%g' % __z )
            ax.set(xlabel = 'r', ylabel = '$\\xi_{%s}(r, z)$'%__type) 
            ax.set_title('Correlation (avg.)', fontdict = {'fontsize': 10, 'fontweight': 'bold',})
            __ypos -= 0.2
            __first = False
        self.exec_time.append([__fname, str(np.shape(x) + (1,)), '%.1f' % __texec])
        return fig
        
def _main():  
    file = p.split(__file__)[1].split('.')[0]
    ###################################################################################     
    #            comparison: class FlatLambdaCDM vs class Cosmology                   #
    ###################################################################################
    m1 = FlatLambdaCDM(0.7, 0.3, 0.05)
    m2 = Cosmology(0.7, 0.3, 0.05) # corresponding `Cosmology` instance
    z  = np.linspace(0, 10, 11)
    print_report([comparison_test(m1.comovingDistance, m2.comovingDistance, z, (0, ), 
                                  title = 'Comoving distance (unit: $Mpc/h$)', 
                                  text = ('$x(a) ~=~ \\frac{c}{H_0} \\int_a^\\infty a^{-2} E^{-1}(a) da$' \
                                          '$~=~ \\frac{2c a^{1/2}}{H_0\\Omega_m^{1/2}} '                  \
                                          '_2F_1(\\frac{1}{6}, \\frac{1}{2}, \\frac{7}{6}, '              \
                                          '-\\frac{\\Omega_{de}}{\\Omega_m}a^3)$; '                       \
                                          '$a = (z+1)^{-1}$'                                              ), 
                                  xlabel = '$z$', 
                                  ylabel = '$x(z)$', 
                                  label1 = 'Analytic',
                                  label2 = 'Numerical', ),
                  comparison_test(m1.time, m2.time, z, (0, ), 
                                  title = 'Time (unit: $H_0^{-1}$)', 
                                  text = ('$t(a) ~=~ \\frac{1}{H_0} \\int_0^a a^{-1} E^{-1}(a) da$' \
                                          '$~=~ \\frac{2 a^{3/2}}{3H_0\\Omega_m^{1/2}} '            \
                                          '_2F_1(\\frac{1}{2}, \\frac{1}{2}, \\frac{3}{2}, '        \
                                          '-\\frac{\\Omega_{de}}{\\Omega_m}a^3)$; '                 \
                                          '$a = (z+1)^{-1}$'                                        ), 
                                  xlabel = '$z$', 
                                  ylabel = '$t(z)$', 
                                  label1 = 'Analytic',
                                  label2 = 'Numerical', ),
                 comparison_test(m1.dplus, m2.dplus, z, (0, ), 
                                  title = 'Linear Growth Factor', 
                                  text  = ('$d_+(a) ~=~ E(a) \\int_0^a a^{-3} E^{-3}(a) da$'   \
                                          '$~=~ \\frac{2 a^{5/2} E(a)}{5H_0\\Omega_m^{3/2}}~' \
                                          '_2F_1(\\frac{5}{6}, \\frac{3}{2}, \\frac{11}{6},'  \
                                          '-\\frac{\\Omega_{de}}{\\Omega_m}a^3)$; '           \
                                          '$a = (z+1)^{-1}$'                                  ), 
                                  xlabel = '$z$', 
                                  ylabel = '$d_+(z)$', 
                                  label1 = 'Analytic',
                                  label2 = 'Numerical', ),
                  comparison_test(m1.dplus, m2.dplus, z, (1, ), 
                                  title = 'Linear Growth Rate', 
                                  text   = '$f(a) ~=~ \\frac{d\\ln d_+}{d\\ln a} $',
                                  xlabel = '$z$', 
                                  ylabel = '$f(z)$', 
                                  label1 = 'Analytic',
                                  label2 = 'Numerical', ), ], 
                path = 'test_output_00.pdf', 
                metadata = {'Title' : 'Comaparison plots', 
                            'Author': file, 
                            'CreationDate': localtime(), }, )
    ###################################################################################     
    #                                  general test                                   #
    ###################################################################################
    print_report(document = GeneralTest().generate_test_report(), 
                 path = 'test_output_01.pdf', 
                 metadata = {'Title': 'Test plots', 
                             'Author': file, 
                             'CreationDate': localtime(), }, )
    return 

if __name__ == '__main__':
    _main()




