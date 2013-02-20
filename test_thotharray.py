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

from thotharray import *
from thoth import *

TEST_FF_TOPO = 'testfiles/flatfile/topo.flat'
TEST_FF_IV = 'testfiles/flatfile/iv.flat'
TEST_FF_MAP = 'testfiles/flatfile/grid.flat'

TEST_INTERACTIVE = False

def array_equal(a, b):
    """Return True if the ndarray a and b are equal."""
    return np.any(np.equal(a, b))


class CheckAxisParam(unittest.TestCase):

    def set_some_value(self):
        """Set some test value in a AxisParam object """
        self.a.length = 9
        self.a.unit = 'm'
        self.a.increment = 0.5
        self.a.start = -2.0
        self.scale = [-2, -1.5, -1, -.5, 0, 0.5, 1, 1.5, 2]

    def setUp(self):
        self.a = AxisParam()
        self.set_some_value()

    def test_set_length(self):
        self.assertIsInstance(self.a.length, int)
        self.a.length = 10
        self.assertEqual(self.a.length, 10)

    def test_set_unit(self):
        self.assertIsInstance(self.a.unit, (unicode, str))
        self.a.unit = u'nm'
        self.assertEqual(self.a.unit, u'nm')

    def test_set_increment(self):
        self.assertIsInstance(self.a.increment, (float, int))
        self.a.increment = 0.8
        self.assertEqual(self.a.increment, 0.8)

    def test_set_start(self):
        self.assertIsInstance(self.a.start, (float, int))
        self.a.start = -3.0
        self.assertEqual(self.a.start,-3.0)

    def test_get_length(self):
        self.assertIsInstance(self.a.length, int)
        self.assertIsInstance(self.a.get_length(), int)
        self.assertIsInstance(len(self.a), int)
        self.set_some_value()
        self.assertEqual(self.a.get_length(), 9)
        self.assertEqual(len(self.a), 9)

    def test_get_physical_length(self):
        self.assertIsInstance(self.a.get_physical_length(), (int, float))

    def test_get_scale(self):
        self.assertTrue(array_equal(self.a.get_scale(), self.scale))

    def test_copy(self):
        b = self.a.copy()
        self.assertIsNot(self.a,b)
        self.assertEqual(self.a.length,b.length)
        self.assertEqual(self.a.unit,b.unit)
        self.assertEqual(self.a.increment,b.increment)
        self.assertEqual(self.a.start,b.start)

    def test_getitem(self):
        for key, type_ in {'length':int,
                           'unit':(str, unicode),
                           'start':(int, float),
                           'increment':(int, float),
                           'scale':(list, np.ndarray),
                           'physical length':(int, float)}.items():
            self.assertIsInstance(self.a[key], type_)

    def test_getitem_wrongkey(self):
        self.assertRaises(KeyError, self.a.__getitem__, 'notavailablekey')

    def test_setitem(self):
        self.a['length'] = 10
        self.assertEqual(self.a.get_length(), 10)

    def test_setitem_notallowed(self):
        self.assertRaises(TypeError, self.a.__setitem__, 'scale', range(9))

class CheckScanningParam(unittest.TestCase):

    def setUp(self):
        pass

    def test_creation(self):
        p = ScanningParam()

    def test_available_parameters(self):
        self.assertIsInstance(ScanningParam().filename, (unicode, str))
        self.assertIsInstance(ScanningParam().type, (unicode, str))
        self.assertIn(ScanningParam().type,
                    ('topo', 'ivcurve', 'ivmap', 'didvcurve',
                     'didvmap', 'topofft', 'ivmapfft', 'didvmapfft'))
        self.assertIsInstance(ScanningParam().direction, (unicode, str))
        self.assertIsInstance(ScanningParam().current, (float, int))
        self.assertIsInstance(ScanningParam().vgap, (float, int))
        self.assertIsInstance(ScanningParam().comment, (unicode, str))
        self.assertIsInstance(ScanningParam().processing, (unicode, str))
        self.assertIsInstance(ScanningParam().unit, (unicode, str))
        self.assertIsInstance(ScanningParam().creation_date, (unicode, str))
        self.assertIsInstance(ScanningParam().comment, (unicode, str))
        self.assertIsInstance(ScanningParam().axis1, AxisParam)
        self.assertIsInstance(ScanningParam().axis2, AxisParam)
        self.assertIsInstance(ScanningParam().axis3, AxisParam)
        self.assertIsInstance(ScanningParam().metadata, dict)

    def test_copy(self):
        p = ScanningParam()
        p.filename = 'test'
        p.current = 10
        q = p.copy()
        self.assertIsNot(p,q)
        self.assertEqual(p.filename,q.filename)
        self.assertEqual(p.current,q.current)

    def test_getitem(self):
        p = ScanningParam()
        for key, type_ in {'type':(str, unicode),
                           'unit':(str, unicode),
                           'direction':(int, float),
                           'current':(int, float),
                           'vgap':(int, float),
                           'processing':(str, unicode),
                           'comment':(str, unicode),
                           'direction':(str, unicode),
                           'axis1':AxisParam,
                           }.items():
            self.assertIsInstance(p[key], type_)

    def test_getitem_wrongkey(self):
        p = ScanningParam()
        self.assertRaises(KeyError, p.__getitem__, 'notavailablekey')

    def test_setitem(self):
        p = ScanningParam()
        p['current'] = 10
        self.assertEqual(p.current, 10)

class CheckMeasurement(unittest.TestCase):

    def setUp(self):
        pass

    def test_available_parameters(self):
        self.assertIsInstance(Measurement().rawdata, np.ndarray)
        self.assertIsInstance(Measurement().param, ScanningParam)

    def test_creation(self):
        a = np.random.rand(5,5)
        p = ScanningParam()
        m1 = Measurement()
        m2 = Measurement(a,p)
        self.assertIs(m2.rawdata, a)
        self.assertIs(m2.param, p)
        m3 = Measurement(rawdata=a)
        self.assertIs(m3.rawdata, a)
        m4 = Measurement(param=p)
        self.assertIs(m4.param, p)

    def test_set_rawdata_from_array(self):
        m = Measurement()
        a = np.random.rand(5,5)
        m.set_rawdata(a)
        self.assertIs(m.rawdata,a)

    def test_set_rawdata_from_measurement(self):
        m1 = Measurement()
        m1.set_rawdata(np.random.rand(5,5))
        m2 = Measurement()
        m2.set_rawdata(m1)
        self.assertIs(m1.rawdata, m2.rawdata)

    def test_set_param_from_param(self):
        m = Measurement()
        s = ScanningParam()
        m.set_param(s)
        self.assertIs(m.param,s)

    def test_copy_rawdata_from_array(self):
        m = Measurement()
        a = np.random.rand(5,5)
        m.copy_rawdata(a)
        self.assertIsNot(m.rawdata,a)
        self.assertTrue(array_equal(m.rawdata, a))

    def test_copy_rawdata_from_measurement(self):
        m1 = Measurement()
        m1.set_rawdata(np.random.rand(5,5))
        m2 = Measurement()
        m2.copy_rawdata(m1)
        self.assertIsNot(m1.rawdata, m2.rawdata)
        self.assertTrue(array_equal(m1.rawdata, m2.rawdata))

    def test_copy_param_from_param(self):
        s = ScanningParam()
        s.current = 5
        m = Measurement()
        m.copy_param(s)
        self.assertIsNot(m.param,s)
        self.assertEqual(m.param.current, s.current)

    def test_copy_param_from_measurement(self):
        s = ScanningParam()
        s.current = 5
        m1 = Measurement()
        m1.set_param(s)
        m2 = Measurement()
        m2.copy_param(m1)
        self.assertIsNot(m1.param, m2.param)
        self.assertEqual(m1.param.current, m2.param.current)

    def test_get_dimension(self):
        m = Measurement(np.random.rand(5))
        self.assertEqual(m.get_dimension(),1)
        m = Measurement(np.random.rand(5,5))
        self.assertEqual(m.get_dimension(), 2)
        m = Measurement(np.random.rand(5,5,5))
        self.assertEqual(m.get_dimension(), 3)

    def test_get_scale(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        correctScale = [-2, -1.5, -1, -.5, 0, 0.5, 1, 1.5, 2]
        s.axis1 = a
        m = Measurement(np.random.rand(5), s)
        self.assertTrue(array_equal(m.get_scale(), correctScale))
        s = ScanningParam()
        s.axis2 = a
        m = Measurement(np.random.rand(5,5), s)
        self.assertTrue(array_equal(m.get_scale(), correctScale))
        s = ScanningParam()
        s.axis3 = a
        m = Measurement(np.random.rand(5,5,5), s)
        self.assertTrue(array_equal(m.get_scale(), correctScale))

    def test_addition_float(self):
        s = ScanningParam()
        s.filename = 'test'
        m1 = Measurement(np.random.rand(5,5),s)
        for value in (0.5, 10, -0.5, -1, -2.3):
            m2 = m1 + value
            self.assertTrue(array_equal(m2.rawdata,
                                        m1.rawdata + value))
            self.assertIsNot(m2.param, m1.param)
            self.assertEqual(m2.param.filename,
                             m1.param.filename)
            self.assertEqual(m2.param.current,
                             m1.param.current)
            self.assertNotEqual(m2.param.processing,
                                m1.param.processing)

    def test_addition(self):
        s1 = ScanningParam()
        s1.filename = 'test'
        s1.current = 2.0
        s1.vgap = -3.0
        s2 = ScanningParam()
        s2.filename = 'other'
        s2.current = 3.0
        s2.vgap = -2.8
        m1 = Measurement(np.random.rand(5,5),s1)
        m2 = Measurement(np.random.rand(5,5),s2)
        m3 = m1 + m2
        m4 = m2 + m1
        self.assertTrue(array_equal(m3.rawdata,
                                    m1.rawdata
                                    + m2.rawdata))
        self.assertIsNot(m3.param, m1.param)
        self.assertIsNot(m3.param, m2.param)
        self.assertNotEqual(m3.param.processing,
                            m1.param.processing)
        self.assertNotEqual(m3.param.processing,
                            m2.param.processing)
        self.assertTrue(array_equal(m3.rawdata,
                                    m4.rawdata))

    def test_addition_wrong_types(self):
        m1 = Measurement(np.random.rand(5))
        m2 = Measurement(np.random.rand(5,5))
        self.assertRaises(ValueError, m1.__add__, m2)

        m1 = Measurement(np.random.rand(6,5))
        m2 = Measurement(np.random.rand(5,5))
        self.assertRaises(ValueError, m1.__add__, m2)

        m1 = 'test'
        m2 = Measurement(np.random.rand(5,5))
        self.assertRaises(TypeError, m1.__add__, m2)

    def test_multiplication(self):
        m1 = Measurement(np.random.rand(5))
        for value in (0.5, 10, -0.5, -1):
            m2 = m1 * value
            self.assertTrue(array_equal(m2.rawdata,
                                        m1.rawdata * value))
            self.assertIsNot(m2.param, m1.param)
            self.assertEqual(m2.param.filename,
                             m1.param.filename)
            self.assertEqual(m2.param.current,
                             m1.param.current)
            self.assertNotEqual(m2.param.processing,
                                m1.param.processing)

    def test_multiplication_wrong_types(self):
        m = Measurement(np.random.rand(5,5))
        for value in ('test', Measurement(np.random.rand(5,5)) ):
            self.assertRaises(TypeError, Measurement.__mul__, m, value)

    def test_division(self):
        m1 = Measurement(np.random.rand(5))
        for value in (0.5, 10, -0.5, -1):
            m2 = m1 / value
            self.assertTrue(array_equal(m2.rawdata,
                                        m1.rawdata / value))
            self.assertIsNot(m2.param, m1.param)
            self.assertEqual(m2.param.filename,
                             m1.param.filename)
            self.assertEqual(m2.param.current,
                             m1.param.current)
            self.assertNotEqual(m2.param.processing,
                                m1.param.processing)

    def test_division_wrong_types(self):
        m = Measurement(np.random.rand(5,5))
        for value in ('test', Measurement(np.random.rand(5,5)) ):
            self.assertRaises(TypeError, Measurement.__div__, m, value)

    def test_getitem(self):
        m = Measurement()
        for key, type_ in {'type':(str, unicode),
                           'unit':(str, unicode),
                           'direction':(int, float),
                           'current':(int, float),
                           'vgap':(int, float),
                           'processing':(str, unicode),
                           'comment':(str, unicode),
                           'direction':(str, unicode),
                           'axis1':AxisParam,
                           }.items():
            self.assertIsInstance(m[key], type_)

    def test_getitem_wrongkey(self):
        m = Measurement()
        self.assertRaises(KeyError, m.__getitem__, 'notavailablekey')

    def test_setitem(self):
        m = Measurement()
        m['current'] = 10
        self.assertEqual(m.param['current'], 10)

    def test_get_shape(self):
        m = Measurement(np.random.rand(5))
        self.assertEqual((5,), m.get_shape())
        m = Measurement(np.random.rand(5, 7))
        self.assertEqual((5, 7), m.get_shape())
        m = Measurement(np.random.rand(5, 7, 6))
        self.assertEqual((5, 7, 6), m.get_shape())

    def test_get_length(self):
        m = Measurement(np.random.rand(5))
        self.assertEqual(5, len(m))
        m = Measurement(np.random.rand(5, 7))
        self.assertEqual(5 * 7, len(m))
        m = Measurement(np.random.rand(5, 7, 6))
        self.assertEqual(6, len(m))

class CheckThothCurveItem(unittest.TestCase):

    def test_creation(self):
        m = Measurement(np.random.rand(5))
        curve1 = ThothCurveItem(m)
        self.assertIs(curve1.measurement,m)
        curve2 = ThothCurveItem()

    def test_set_measurement(self):
        curve = ThothCurveItem()
        m = Measurement(np.random.rand(5))
        curve.set_measurement(m)
        self.assertIs(curve.measurement,m)

    def test_set_wrong_measurement(self):
        curve = ThothCurveItem()
        m1 = Measurement(np.random.rand(5,5))
        m2 = Measurement(np.random.rand(5,5,5))
        for m in (m1, m2, 'anything', 5, AxisParam()):
            self.assertRaises(AssertionError, curve.set_measurement, m)

    def test_check_data(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        s.axis1 = a
        data = np.arange(9)
        m = Measurement(data,s)
        curve = ThothCurveItem(m)
        x,y = curve.get_data()
        self.assertTrue(array_equal(data,y))
        self.assertTrue(array_equal(m.get_scale(),x))

    def test_adding_to_plot(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        s.axis1 = a
        m = Measurement(np.arange(9),s)
        curve = ThothCurveItem(m)
        from guiqwt.curve import CurvePlot
        plot = CurvePlot(title="Test")
        plot.add_item(curve)
        plot.show()

    def test_addition(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        s.axis1 = a
        m = Measurement(np.arange(9),s)
        curve = ThothCurveItem(m)
        curve2 = curve + curve
        self.assertIsInstance(curve2, ThothCurveItem)

    def test_multiplication(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        s.axis1 = a
        m = Measurement(np.arange(9),s)
        curve = ThothCurveItem(m)
        curve2 = curve * 2
        self.assertIsInstance(curve2, ThothCurveItem)

    def test_division(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        s.axis1 = a
        m = Measurement(np.arange(9),s)
        curve = ThothCurveItem(m)
        curve2 = curve / 2
        self.assertIsInstance(curve2, ThothCurveItem)

    def test_getitem(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        s.axis1 = a
        s.vgap = 2.8
        data = np.arange(9)
        m = Measurement(data,s)
        curve = ThothCurveItem(m)
        vgap = curve['vgap']
        self.assertEqual(2.8, vgap)
        data_item = curve['rawdata']
        self.assertIsInstance(data_item, np.ndarray)
        self.assertTrue(array_equal(data, data_item ))
        self.assertRaises(KeyError, curve.__getitem__, 'notavailablekey')
        self.assertRaises(TypeError, curve.__getitem__, 1)

    def test_setitem(self):
        curve = ThothCurveItem()
        curve['vgap'] = 3.
        self.assertEqual(3, curve['vgap'])

    def test_compute(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        s.axis1 = a
        s.vgap = 2.8
        data = np.arange(9)
        m = Measurement(data,s)
        curve = ThothCurveItem(m)
        new_curve = curve.compute('Identity', lambda x: x)
        self.assertIsNot(new_curve, curve)
        self.assertTrue(array_equal(new_curve['rawdata'], curve['rawdata']))
        class TestParam(DataSet):
            value = FloatItem('coefficient', default=1)
        param = TestParam()
        param.value = 3.5
        new_curve = curve.compute('Multiply', lambda x, p: x * p.value,
                                  param=param,
                                  suffix=lambda p: 'multiplied by %.3f' % \
                                                    p.value,
                                  interactive=False)
        if TEST_INTERACTIVE:
            new_curve = curve.compute('Multiply', lambda x, p: x * p.value,
                                    param=param,
                                    suffix=lambda p: 'multiplied by %.3f' % \
                                                        p.value,
                                    interactive=True)
            new_curve = curve.compute('Multiply', lambda x, p: x * p.value,
                                    param=param,
                                    suffix=lambda p: 'multiplied by %.3f' % \
                                                        p.value,
                                    interactive='text')

    def create_curve(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 9
        a.increment = 0.5
        a.start = -2.0
        a.unit = 'V'
        s.axis1 = a
        s.vgap = 2.8
        data = 3 * np.arange(5) + np.random.rand(5)
        m = Measurement(data,s)
        return ThothCurveItem(m)

    def test_computation_function(self):
        curve = self.create_curve()
        for func in (curve.compute_detrend,
                     curve.compute_derivative,
                     curve.compute_fft,
                     curve.compute_ifft,
                     curve.compute_wiener,
                     curve.compute_zeroes,
                     ):
            result = func()
            self.assertIsNot(curve, result)
            self.assertIsInstance(result, ThothCurveItem)

        for func in (curve.compute_savitzky,
                     curve.compute_gaussian,
                     ):
            result = func(interactive=False)
            self.assertIsNot(curve, result)
            self.assertIsInstance(result, ThothCurveItem)

class CheckThothImageItem(unittest.TestCase):

    def test_creation(self):
        m = Measurement(np.random.rand(5,5))
        image1 = ThothImageItem(m)
        self.assertIs(image1.measurement,m)
        image2 = ThothImageItem()

    def test_set_measurement(self):
        image = ThothImageItem()
        m = Measurement(np.random.rand(5,5))
        image.set_measurement(m)
        self.assertIs(image.measurement,m)

    def test_set_wrong_measurement(self):
        image = ThothImageItem()
        m1 = Measurement(np.random.rand(5))
        m2 = Measurement(np.random.rand(5,5,5))
        for m in (m1, m2, 'anything', 5, AxisParam()):
            self.assertRaises(AssertionError, image.set_measurement, m)

    def test_check_data(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 5
        a.increment = 0.5
        a.start = -1.5
        a.unit = 'nm'
        s.axis1 = a
        s.axis2 = a
        data = np.random.rand(5,5)
        m = Measurement(data,s)
        image = ThothImageItem(m)
        self.assertTrue(array_equal(image.data,data))

    def test_adding_to_plot(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 5
        a.increment = 0.5
        a.start = -1.5
        a.unit = 'nm'
        s.axis1 = a
        s.axis2 = a
        data = np.random.rand(5,5)
        m = Measurement(data,s)
        image = ThothImageItem(m)

        from guiqwt.image import ImagePlot
        plot = ImagePlot(title="Test")
        plot.add_item(image)
        plot.show()

    def test_addition(self):
        m = Measurement(np.random.rand(5,5))
        image = ThothImageItem(m)
        image2 = image + image
        self.assertIsInstance(image2, ThothImageItem)
        self.assertTrue(array_equal(m.rawdata * 2, image2['rawdata']))
        image2 = image + 5.
        self.assertIsInstance(image2, ThothImageItem)
        self.assertTrue(array_equal(m.rawdata + 5, image2['rawdata']))

    def test_multiplication(self):
        m = Measurement(np.random.rand(5,5))
        image = ThothImageItem(m)
        image2 = image * 2.
        self.assertIsInstance(image2, ThothImageItem)

    def test_division(self):
        m = Measurement(np.random.rand(5,5))
        image = ThothImageItem(m)
        image2 = image / 2.
        self.assertIsInstance(image2, ThothImageItem)

    def create_image(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 5
        a.increment = 0.5
        a.start = -1.5
        a.unit = 'nm'
        s.axis1 = a
        s.axis2 = a
        s.vgap = 2.8
        data = np.random.rand(5,5)
        m = Measurement(data,s)
        return ThothImageItem(m)

    def test_computation_function(self):
        image = self.create_image()
        for func in (image.compute_line_detrend,
                     image.compute_fft,
                     image.compute_ifft,
                     image.compute_wiener,
                     image.compute_zeroes,
                     ):
            result = func()
            self.assertIsNot(image, result)
            self.assertIsInstance(result, ThothImageItem)

        for func in (image.compute_savitzky,
                     image.compute_gaussian,
                     ):
            result = func(interactive=False)
            self.assertIsNot(image, result)
            self.assertIsInstance(result, ThothImageItem)

def create_test_map():
    s = ScanningParam()
    a1 = AxisParam()
    a1.length = np.random.randint(5,8)
    a1.increment = 0.5
    a1.start = 0
    a1.unit = 'nm'
    a2 = AxisParam()
    a2.length = np.random.randint(5,8)
    a2.increment = 0.5
    a2.start = 0
    a2.unit = 'nm'
    a3 = AxisParam()
    a3.length = np.random.randint(5,8)
    a3.increment = 0.2
    a3.start = -3.0
    a3.unit = 'V'
    s.axis1 = a1
    s.axis2 = a2
    s.axis3 = a3
    data = np.random.rand(a3.length,a2.length,a1.length)
    m = Measurement(data,s)
    map_ = ThothMapItem(m)
    return map_

class CheckThothMapItem(unittest.TestCase):

    def test_creation(self):
        m = Measurement(np.random.rand(5,5,5))
        map1 = ThothMapItem(m)
        self.assertIs(map1.measurement,m)
        map2 = ThothMapItem()

    def test_set_measurement(self):
        map_ = ThothMapItem()
        m = Measurement(np.random.rand(5,5,5))
        map_.set_measurement(m)
        self.assertIs(map_.measurement,m)

    def test_set_wrong_measurement(self):
        map_ = ThothMapItem()
        m1 = Measurement(np.random.rand(5))
        m2 = Measurement(np.random.rand(5,5))
        for m in (m1, m2, 'anything', 5, AxisParam()):
            self.assertRaises(AssertionError, map_.set_measurement, m)

    def test_check_data(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 5
        a.increment = 0.5
        a.start = -1.5
        a.unit = 'nm'
        s.axis1 = a
        s.axis2 = a
        s.axis3 = a
        data = np.random.rand(5,5,5)
        m = Measurement(data,s)
        map_ = ThothMapItem(m)
        self.assertTrue(array_equal(map_.data,data))

    def test_adding_to_plot(self):
        s = ScanningParam()
        a = AxisParam()
        a.length = 5
        a.increment = 0.5
        a.start = -1.5
        a.unit = 'nm'
        s.axis1 = a
        s.axis2 = a
        s.axis3 = a
        data = np.random.rand(5,5,5)
        m = Measurement(data,s)
        map_ = ThothMapItem(m)

        from guiqwt.image import ImagePlot
        plot = ImagePlot(title="Test")
        plot.add_item(map_)
        plot.show()

    def test_set_slice_index_valid_input(self):
        map_ = create_test_map()
        self.assertEqual(0, map_.get_current_index())
        index_max = map_.measurement.param.axis3.length
        data = map_.measurement.rawdata
        map_.set_slice_index(index_max-1)
        self.assertEqual(index_max-1, map_.get_current_index())
        self.assertTrue(array_equal(map_.data,data[-1,:,:]))
        map_.set_slice_index(1)
        self.assertEqual(1, map_.get_current_index())
        self.assertTrue(array_equal(map_.data,data[1,:,:]))
        map_.set_slice_index(3)
        self.assertEqual(3, map_.get_current_index())
        self.assertTrue(array_equal(map_.data,data[3,:,:]))
        map_.set_slice_index(-3)
        self.assertEqual(index_max-3, map_.get_current_index())
        self.assertTrue(array_equal(map_.data,data[-3,:,:]))
        map_.set_slice_index(0)
        self.assertEqual(0, map_.get_current_index())
        self.assertTrue(array_equal(map_.data,data[0,:,:]))

    def test_set_silce_index_invalid_input(self):
        map_ = create_test_map()
        for index in (3.3, 'string', None):
            self.assertRaises(AssertionError, map_.set_slice_index, index)

    def test_set_slice_index_out_of_bounds(self):
        map_ = create_test_map()
        index_max = map_.measurement.param.axis3.length
        for index in (index_max+1, -index_max-1, index_max+10):
            self.assertRaises(IndexError, map_.set_slice_index, index)

    def test_set_slice_value_valid_input(self):
        map_ = create_test_map()
        index_max = map_.measurement.param.axis3.length
        data = map_.measurement.rawdata
        map_.set_slice_value(-2.6)
        self.assertTrue(array_equal(map_.data,data[2,:,:]))
        map_.set_slice_value(-2.5) # interpolated value
        map_.set_slice_value(-2.4)
        self.assertTrue(array_equal(map_.data,data[3,:,:]))

    def test_set_slice_value_invalid_input(self):
        map_ = create_test_map()
        for value in ('string', None):
            self.assertRaises(AssertionError, map_.set_slice_value, value)

    def test_set_slice_value_out_of_bound(self):
        map_ = create_test_map()
        for value in (-4, 0.5, 1000 ):
            self.assertRaises(ValueError, map_.set_slice_value, value)

    def test_get_curve_at_index_valid_pixels(self):
        map_ = create_test_map()
        curve1 = map_.get_curve_at_index(0,1) # one pixel
        self.assertIsInstance(curve1, ThothCurveItem)
        self.assertEqual(curve1.measurement.get_dimension(),1)
        self.assertTrue(array_equal(curve1.measurement.param.axis1.get_scale(),
                               map_.measurement.param.axis3.get_scale()))
        self.assertEqual(len(curve1.measurement.rawdata),
                         map_.measurement.param.axis3.get_length())
        curve0 = map_.get_curve_at_index(0,0)
        self.assertIsInstance(curve0, ThothCurveItem)
        curve2 = map_.get_curve_at_index(1,0)
        self.assertIsInstance(curve2, ThothCurveItem)
        curve3 = map_.get_curve_at_index(1,1)
        self.assertIsInstance(curve3, ThothCurveItem)
        averaged = (curve0.measurement + curve1.measurement + \
                    curve2.measurement + curve3.measurement) / 4.
        self.assertFalse(array_equal(curve0['rawdata'],
                                    curve1['rawdata']))
        curve = map_.get_curve_at_index(0,0,1,1) # averaged over pixels
        self.assertIsInstance(curve, ThothCurveItem)
        self.assertEqual(curve.measurement.get_dimension(),1)
        self.assertTrue(array_equal(curve.measurement.param.axis1.get_scale(),
                               map_.measurement.param.axis3.get_scale()))
        self.assertEqual(len(curve.measurement.rawdata),
                         map_.measurement.param.axis3.get_length())
        self.assertTrue(array_equal(averaged.rawdata,
                                    curve.measurement.rawdata))
        for pos in ((1,2,0,0),(2,0,0,0),(0,1,3,3)):
            curve = map_.get_curve_at_index(*pos)
            self.assertIsInstance(curve, ThothCurveItem)

    def test_get_curve_at_index_invalid_pixels(self):
        map_ = create_test_map()
        self.assertRaises(TypeError, map_.get_curve_at_index, 0, 0, 1)
        self.assertRaises(IndexError, map_.get_curve_at_index, 12, 11)
        self.assertRaises(IndexError, map_.get_curve_at_index, 1, 3, 12, 14)
        self.assertRaises(IndexError, map_.get_curve_at_index, 12, 14, 3, 2)
        self.assertRaises(IndexError, map_.get_curve_at_index, -1, -1, 3, 2)

    def test_computation_function(self):
        map_ = create_test_map()
        for func in (map_.compute_fft,
                     map_.compute_ifft,
                     map_.compute_wiener,
                     ):
            result = func()
            self.assertIsNot(map_, result)
            self.assertIsInstance(result, ThothMapItem)

        for func in (map_.compute_savitzky,
                     map_.compute_gaussian,
                     ):
            result = func(interactive=False)
            self.assertIsNot(map_, result)
            self.assertIsInstance(result, ThothMapItem)


class CheckThoth(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_items(self):
        t = Thoth()
        self.assertIsInstance(t.get_items(), list)
        self.assertEqual(0, len(t.get_items()))

    def test_get_windows(self):
        t = Thoth()
        self.assertIsInstance(t.get_windows(), list)
        self.assertEqual(0, len(t.get_windows()))

    def test_register_measurement(self):
        t = Thoth()
        m = Measurement()
        m.param.type = 'wrong type'
        self.assertRaises(ValueError, t.register_measurement, m)
        self.assertEqual(0, len(t.get_items()))
        m = Measurement(np.random.rand(5))
        m.param.type = 'ivcurve'
        t.register_measurement(m)
        self.assertEqual(1, len(t.get_items()))
        m = Measurement(np.random.rand(5,5))
        m.param.type = 'topo'
        t.register_measurement(m)
        self.assertEqual(2, len(t.get_items()))
        m = Measurement(np.random.rand(5,5,5))
        m.param.type = 'ivmap'
        t.register_measurement(m)
        self.assertEqual(3, len(t.get_items()))
        t.register_measurement((m,m))
        self.assertEqual(5, len(t.get_items()))

    def test_register_item(self):
        t = Thoth()
        map_ = ThothMapItem()
        t.register_item(map_)
        self.assertEqual(1, len(t.get_items()))
        t.register_item((map_, map_))
        self.assertEqual(3, len(t.get_items()))

    def test_open_dialog(self):
        if TEST_INTERACTIVE:
            t = Thoth()
            t.open()

    def test_open_flatfile_one_file(self):
        t = Thoth()
        t.open(TEST_FF_TOPO)
        self.assertEqual(2, len(t.get_items()))

    def test_open_flatfile_many_files(self):
        t = Thoth()
        t.open(TEST_FF_TOPO, TEST_FF_IV) # many files name
        self.assertEqual(3, len(t.get_items()))
        t.open((TEST_FF_TOPO, TEST_FF_IV)) # a list
        self.assertEqual(6, len(t.get_items()))

    def test_create_window(self):
        t = Thoth()
        m = Measurement(np.random.rand(5))
        m.param.type = 'ivcurve'
        t.register_measurement(m)
        t.create_window(t.get_items()[0])
        self.assertEqual(1, len(t.get_windows()))
        m = Measurement(np.random.rand(5,5))
        m.param.type = 'topo'
        t.register_measurement(m)
        t.create_window(t.get_items()[1])
        self.assertEqual(2, len(t.get_windows()))
        map_ = create_test_map()
        t.register_item(map_)
        item = t.get_items()[-1]
        t.create_window(item)
        self.assertEqual(3, len(t.get_windows()))
        t.create_window((item, item))
        self.assertEqual(5, len(t.get_windows()))

def graphical_inspection_curve():
    s = ScanningParam()
    a = AxisParam()
    a.length = 9
    a.increment = 0.5
    a.start = -2.0
    s.axis1 = a
    m = Measurement(arange(9),s)
    curve = ThothCurveItem(m)

    from guiqwt.plot import CurveWindow
    win = CurveWindow(wintitle="Test")
    plot = win.get_plot()
    plot.add_item(curve)
    win.show()
    return win

def graphical_inspection_image():
    s = ScanningParam()
    a = AxisParam()
    a.length = 5
    a.increment = 0.5
    a.start = -1.5
    a.unit = 'nm'
    s.axis1 = a
    s.axis2 = a
    data = np.random.rand(5,5)
    m = Measurement(data,s)
    image = ThothImageItem(m)

    from guiqwt.plot import ImageWindow
    win = ImageWindow(wintitle="Test")
    plot = win.get_plot()
    plot.add_item(image)
    win.show()
    return win

def graphical_inspection_map():
    map_ = create_test_map()

    from guiqwt.plot import ImageWindow
    win = ImageWindow(wintitle="Test")
    plot = win.get_plot()
    plot.add_item(map_)
    win.show()
    return win

def graphical_test():
    graphical_inspection_curve()
    graphical_inspection_image()
    graphical_inspection_map()

if __name__ == '__main__':
    unittest.main()

