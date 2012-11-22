# -*- coding: utf-8 -*-

"""
    Copyright © 2011 François Bianco, University of Geneva - francois.bianco@unige.ch

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import numpy as np

from guidata import qapplication
_app = qapplication()

from flatfileproxy import *

TEST_FF_TOPO = 'testfiles/flatfile/topo.flat'
TEST_FF_IV = 'testfiles/flatfile/iv.flat'
TEST_FF_MAP = 'testfiles/flatfile/grid.flat'

class CheckFlatFileProxy(unittest.TestCase):

    def setUp(self):
        ffp = None

    def test_get_measurements(self):
        ffp = FlatFileProxy()
        self.assertIsInstance(ffp.get_measurements(), list)

    def test_open_file(self):
        ffp = FlatFileProxy()
        ffp.open(TEST_FF_TOPO)
        measurements = ffp.get_measurements()
        self.assertEqual(2, len(measurements))
        for m in measurements:
            self.assertIsInstance(m, Measurement)
            self.assertGreater(len(m.rawdata),0)

    def test_parameters_curve(self):
        ffp = FlatFileProxy()
        ffp.open(TEST_FF_IV)
        m = ffp.get_measurements()[0]
        self.assertEqual('ivcurve', m.param.type)
        self.assertEqual(m.param.axis1.length, len(m.rawdata))
        self.assertNotEqual('', m.param.direction)

    def test_parameters_topo(self):
        ffp = FlatFileProxy()
        ffp.open(TEST_FF_TOPO)
        m = ffp.get_measurements()[0]
        self.assertEqual('topo', m.param.type)
        sizey, sizex = m.rawdata.shape
        self.assertEqual(sizex, m.param.axis1.length)
        self.assertEqual(sizey, m.param.axis2.length)
        self.assertNotEqual('', m.param.direction)


    def test_parameters_map(self):
        ffp = FlatFileProxy()
        ffp.open(TEST_FF_MAP)
        m = ffp.get_measurements()[0]
        self.assertEqual('ivmap', m.param.type)
        sizez, sizey, sizex = m.rawdata.shape
        self.assertEqual(sizex, m.param.axis1.length)
        self.assertEqual(sizey, m.param.axis2.length)
        self.assertEqual(sizez, m.param.axis3.length)
        self.assertNotEqual('', m.param.direction)



if __name__ == '__main__':
    unittest.main()