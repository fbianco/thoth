# -*- coding: utf-8 -*-

"""
    Copyright © 2011 François Bianco, University of Geneva - francois.bianco@unige.ch

    You may copy this example for any purpose without any restrictions.
"""

import sys
from os import listdir, chdir
import os
from thoth import Thoth

PREVIEW=False

def main():
    # create measurement manager
    thoth = Thoth()

    items = []
    for path in sys.argv[1:]:
        
        # get list of files in example dir
        try:
            filenames = listdir(path)
        except OSError:
            print "Unable to find folder, ignore."
            continue

        # open all the files
        chdir(path)
        thoth.open(filenames)

        # get the average, take smoothed derivative
        # smooth result, display it and try to fit bcs
        average = thoth.get_average()

        smoothed = average.compute_gaussian(interactive=False)

        class Param:
            num_points = 4
            poly_degree = 3
            diff_order = 1
        param = Param()
        didv = smoothed.compute_savitzky(param, interactive=False)
        thoth.create_window(didv)
        if PREVIEW:
            thoth.create_window(average)
            didv_smoothed = didv.compute_wiener()
            thoth.create_window(didv_smoothed)
            thoth.create_window(didv.compute_gaussian(interactive=False))
            thoth.create_window(smoothed.compute_derivative())
            thoth.create_window(smoothed)

        items.append(didv)

    ## TODO not fully implemented in thoth
    #thoth.create_multi_curves_window(items)

if __name__ == "__main__":
    try :
        from guidata import qapplication
        _app = qapplication()
        main()
        _app.exec_()
    except (KeyboardInterrupt) :
        print "Goodbye world !"
