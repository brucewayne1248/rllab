#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:31:52 2018

@author: andi
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# load results


if __name__=="__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("file", type=str,
                       help="path to the benchmark file")
   args = parser.parse_args()
   save = True


   directory = os.path.dirname(os.path.abspath(args.file))
   benchmark = pickle.load(open(args.file, "rb"))

   dist_relmins = np.array(benchmark["dist_relmins"])
   dist_mins = np.array(benchmark["dist_mins"])
   anglediffs_tangent = np.array(benchmark["anglediffs_tangent"])

#   n = len(dist_mins)
   n = 750
   width = 4
   height = 8

   idx = np.random.choice(np.arange(len(dist_mins)), n, replace=False)

   dist_relmins_samples = dist_relmins[idx]
   dist_mins_samples = dist_mins[idx]
   anglediffs_tangent_samples = anglediffs_tangent[idx]

   plt.figure(figsize=[width,height])
   plt.rcParams["axes.grid.axis"] = "y"
   plt.rcParams["axes.grid"] = True
   plt.boxplot(100*(dist_relmins_samples))
   plt.xlabel("{} Datenpunkte\nMedian: {:.2f}%\nDurchschnitt: {:.2f}%"
              .format(n, 100*np.median(dist_relmins_samples), 100*np.mean(dist_relmins_samples)))
   plt.ylabel("Fehler in %")
   plt.title("Relativer Positionsfehler bezogen auf \ndie Gesamtlänge des Kontinuumsroboters")
   if save: plt.savefig(directory+"/boxplot_dist_rel.svg")

   plt.figure(figsize=[width,height])
   plt.boxplot(1000*(dist_mins_samples))
#   plt.xlabel("r9#75")
   plt.xlabel("{} Datenpunkte\nMedian: {:.2f}mm\nDurchscnitt: {:.2f}mm"
              .format(n, 1000*np.median(dist_mins_samples), 1000*np.mean(dist_mins_samples)))
   plt.ylabel("Abstand in mm")
   plt.title("Absoluter Positionsfehler")
   if save: plt.savefig(directory+"/boxplot_dist_abs.svg")

   plt.figure(figsize=[width,height])
   plt.boxplot(180/np.pi*(anglediffs_tangent_samples))
   plt.xlabel("{} Datenpunkte\nMedian: {:.2f}°\nDurchschnitt {:.2f}°"
              .format(n, 180/np.pi*np.median(anglediffs_tangent_samples), 180/np.pi*np.mean(anglediffs_tangent_samples)))
   plt.title("Orientierungsfehler bezogen\nauf den Tangentenvektor")
   plt.ylabel("Winkel in Grad")
   if save: plt.savefig(directory+"/boxplot_orientierung.svg")

   plt.show()
   plt.pause(0.25)

   print("Boxplots saved under: {}".format(directory))