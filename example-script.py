# -*- coding: utf-8 -*-

"""
    Copyright © 2011 François Bianco, University of Geneva - francois.bianco@unige.ch

    You may copy this example for any purpose without any restrictions.
"""

from os import listdir, chdir

from thoth import Thoth

PREVIEW = True

def main():
    # create measurement manager
    thoth = Thoth()

    # get list of files in example dir
    chdir('example')
    filenames = listdir('.')

    # open all the files
    thoth.open(filenames)

    # get the average, take smoothed derivative
    # smooth result, display it and try to fit bcs
    average = thoth.get_average()

    class Param:
        num_points = 8
        poly_degree = 4
        diff_order = 1
    param = Param()
    didv = average.compute_savitzky(param, interactive=False)
    didv_smoothed = didv.compute_wiener()

    if PREVIEW:
        thoth.create_window(didv_smoothed)
        thoth.create_window(average)
    fit = didv_smoothed.fit_bcs()
    print fit.get_values()

if __name__ == "__main__":
    try :
        from guidata import qapplication
        _app = qapplication()
        main()
        _app.exec_()
    except (KeyboardInterrupt) :
        print "Goodbye world !"