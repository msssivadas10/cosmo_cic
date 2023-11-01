#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from cic.models.cosmology import Cosmology, plank18
from cic.models.halos.density.cmrelations import bullock01, zheng07
from cic.models.halos.density.profiles import NFW


nfw = NFW(bullock01) # nfw halo profile calculator

plt_args = dict(marker = 's', ls = '-', ms = 4) # common plot settings

# object to store data and plot options
@dataclass
class PlotData:
    x: np.ndarray
    y: np.ndarray
    xlabel: str       = None
    ylabel: str       = None
    title: str        = None
    logx: bool        = False
    logy: bool        = False
    legend: list      = None
    legend_title: str = ''
    xlim: tuple       = None
    ylim: tuple       = None
    cm: object        = None

    # return the x and y data
    def data(self): return self.x, self.y

    # create a plot with available data
    def plot(self):

        # creating figure and axis
        fig = plt.figure()
        ax  = fig.add_axes([0.12, 0.12, 0.8, 0.7])

        # set axis scale
        if self.logx and self.logy: ax.loglog()
        elif self.logx: ax.semilogx()
        elif self.logy: ax.semilogy()

        # plot data
        plt.plot(self.x, self.y, **plt_args)

        if self.legend is not None: ax.legend(self.legend, title = self.legend_title) # create legend
        if self.xlabel is not None: ax.set_xlabel(self.xlabel) # create x label
        if self.ylabel is not None: ax.set_ylabel(self.ylabel) # create y label
        if self.title is not None: # create title
            title = ' '.join([x.capitalize() for x in self.title.split(' ')])
            ax.set_title(title, y = 1.15)
        if self.cm is not None: # create parameter table
            cm = self.cm
            ax.table(colLabels = ['$\\Omega_m$', '$\\Omega_b$', '$\\Omega_{de}$', '$n_s$', '$\\sigma_8$'], 
                     cellText = [list(map(str, [cm.Om0, cm.Ob0, cm.Ode0, cm.ns, cm.sigma8]))], 
                     bbox = [0., 1., 1., 0.15], zorder = 10, colColours = ['#eee'] * 5, )
        if self.xlim is not None: ax.set_xlim(self.xlim) # set x limit
        if self.ylim is not None: ax.set_ylim(self.ylim) # set y limit
        plt.show()
        return self

# decorator to plot the results
def plot(func):
    def wrapper(*args, **kwargs):
        pd = func(*args, **kwargs).plot()
        return pd
    return wrapper

# create a test cosmology model
def testCosmology():
    cm = Cosmology(h = 0.7, Om0 = 0.3, Ob0 = 0.05, sigma8 = 0.8, name = 'test_cosmology')
    cm.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
    return cm

# matter power spectrum
@plot
def matterPowerspec(cm, z = [0., 2., 5.], start = -3., stop = 3., pts = 21):
    k  = np.logspace(start, stop, pts) 
    pd = PlotData(k, cm.matterPowerSpectrum(k, z, grid = True), 
                  xlabel = 'k', ylabel = 'P(k, z)', title = 'power spectrum', 
                  logx = True, logy = True, legend = z, legend_title = 'redshift, z',
                  cm = cm)
    return pd

# matter variance
@plot
def matterVariance(cm, z = [0., 2., 5.], start = -2., stop = 2., pts = 21):
    r = np.logspace(start, stop, pts) 
    pd = PlotData(r, cm.matterVariance(r, z, grid = True), 
                  xlabel = 'r', ylabel = '$\\sigma^2$(k, z)', title = 'variance', 
                  logx = True, logy = True, legend = z, legend_title = 'redshift, z',
                  cm = cm)
    return pd

# halo mass function
@plot
def haloMassfunction(cm, z = [0., 2., 5.], start = 6., stop = 14., pts = 21):
    m = np.logspace(start, stop, pts)
    pd = PlotData(m, cm.massFunction(m, z, grid = True), 
                  xlabel = 'm', ylabel = 'dn(m, z)/dlnm', title = 'mass function', 
                  logx = True, logy = True, legend = z, legend_title = 'redshift, z',
                  cm = cm)
    return pd

# halo bias
@plot
def haloBias(cm, z = [0., 2., 5.], start = 6., stop = 14., pts = 21):
    m = np.logspace(start, stop, pts)
    pd = PlotData(m, np.log( cm.biasFunction(m, z, grid = True) ), 
                  xlabel = 'm', ylabel = 'ln b(m, z)', title = 'linear bias', 
                  logx = True, logy = False, legend = z, legend_title = 'redshift, z',
                  cm = cm)
    return pd

# linear growth factor
@plot
def growthFactor(cm, start = 0., stop = 5., pts = 21):
    z = np.linspace(start, stop, pts)
    pd = PlotData(z, cm.dplus(z, 0) / cm.dplus(0), 
                  xlabel = 'z', ylabel = '$D_+(z)$', title = 'linear growth',
                  cm = cm)
    return pd

# angular diameter distance
@plot
def angularDistance(cm, start = 0., stop = 5., pts = 21):
    z = np.linspace(start, stop, pts)
    pd = PlotData(z, cm.angularDiameterDistance(z), 
                  xlabel = 'z', ylabel = '$d_A$(z)', title = 'angular dimeter distance',
                  cm = cm)
    return pd

# nfw halo profile
@plot
def haloProfile(cm, m = 1e+13, z = [0., 2., 5.], start = -3., stop = 3., pts = 21):
    x = np.logspace(start, stop, pts)
    pd = PlotData(x, nfw(cm, x, m, z, 200., ft = True, truncate = False, grid = False), 
                  xlabel = 'k', ylabel = '$u_{NFW}(k|m,z)$', title = 'halo profile',
                  logx = True, logy = False, legend = z, legend_title = f'mass, m = {m:g}\nredshift, z:',
                  cm = cm)
    return pd

def testAll(cm, what = 'pvmbgdu'):

    # creating interpolation tables
    cm.createInterpolationTables()

    # normalisation (massfunction, bias etc not work oterwise)
    cm.normalizePowerSpectrum()

    # creating plots
    if 'p' in what: matterPowerspec(cm)
    if 'v' in what: matterVariance(cm)
    if 'm' in what: haloMassfunction(cm)
    if 'b' in what: haloBias(cm)
    if 'g' in what: growthFactor(cm)
    if 'd' in what: angularDistance(cm)
    if 'u' in what: haloProfile(cm)
    return 

if __name__ == '__main__':
    cm = plank18 # testCosmology()
    testAll(cm, 'u')
