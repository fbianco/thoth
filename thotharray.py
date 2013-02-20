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

from __future__ import division

from numpy import arange, ones, array, ndarray, where
from copy import deepcopy

from guiqwt.transitional import QwtPlotItem
from guidata.qt.QtCore import QObject, SIGNAL, Qt
from guidata.qt.QtGui import QApplication, QCursor, QMessageBox
from guidata.dataset.datatypes import (DataSet, BeginTabGroup, EndTabGroup,
                                       ObjectItem)
from guidata.dataset.dataitems import (FloatItem, FloatArrayItem, IntItem,
                                       ChoiceItem, FileOpenItem, StringItem,
                                       TextItem, DictItem)
from guidata.dataset.qtwidgets import DataSetEditLayout, DataSetShowLayout
from guidata.dataset.qtitemwidgets import DataSetWidget
from guidata.utils import update_dataset
from guiqwt.image import CurveItem, ImageItem
from guiqwt.styles import CurveParam, ImageParam
from guiqwt.config import _


APP_NAME = 'Thoth'

class AxisParam(DataSet):
    """Store the parameters to caracterise a measurement axis"""

    unit = StringItem("Physical unit", default='')
    length = IntItem("Number of point", default=0,
            help="Number of measured point along the axis")
    start = FloatItem("Physical start value", default=0)
    increment = FloatItem("Physical increment", default=1,
            help="Physical size of a pixel")

    scale = None # lazy object, created on demand

    def get_length(self):
        """Return the length of the scale"""
        return int(self.length)

    def get_physical_length(self):
        """Return the physical length of the scale
           see self.unit for the unit
        """
        return self.length * self.increment

    def get_scale(self):
        """Return a vector with all the physical axis values based on start,
        length and increment parameters."""
        if self.scale is None : # create lazy object
            self.update_scale()
        return self.scale

    def update_scale(self):
        """Update/create a vector with all the physical axis values based on start, resolution and increment parameters."""

        self.scale = self.start + arange(self.length) * self.increment

    def copy(self):
        """Return a copy of the object"""
        return deepcopy(self)

    def __len__(self):
        return self.get_length()

    def __getitem__(self, key):
        if 'physical length' == key:
            return self.get_physical_length()
        elif 'length' == key:
            return self.get_length()
        elif 'scale' == key:
            return self.get_scale()
        else:
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError, _("They are no such attributes %s.") % key

    def __setitem__(self, key, value):
        if key in ('scale', 'physical length'):
            raise TypeError, _("Not mutable item.")
        setattr(self, key, value)

# Register the new class as a DataSetItem
class AxisParamWidget(DataSetWidget):
    klass = AxisParam
class AxisParamItem(ObjectItem):
    klass = AxisParam
DataSetEditLayout.register(AxisParamItem, AxisParamWidget)
DataSetShowLayout.register(AxisParamItem, AxisParamWidget)


class ScanningParam(DataSet):
    """Store the parameters describing a scanning probe measurement."""

    filename = FileOpenItem(_('File name'), ('*'), default='',
                            help=_('Raw file name'))

    type = ChoiceItem( _('Type of data'),
           [('topo', _('Topography')),
            ('ivcurve', _('IV curve')),
            ('ivmap', _('IV map')),
            ('didvcurve', _('dIdV curve')),
            ('didvmap', _('dIdV map')),
            ('topofft', _('FFT of topography')),
            ('ivmapfft', _('FFT of IV map')),
            ('didvmapfft', _('FFT of dIdV map')),
            ('unknowncurve', _('Curve unknown')),
            ('unknownimage', _('Image unknown')),
            ('unknownmap', _('Map unknown')),
            ('unknown', _('Unknown'))],
           default='topo',
           help='Type of data, determine default color of plots and possible calculation of the data type.')

    direction = StringItem('Direction of mesurement',
            # Possible value, 'up-fwd', 'up-bwd', 'down-fwd', 'down-bwd',
            #                 'fwd', 'bwd', 'up-fwd mirrored', …
            default='',
            help='Direction of measurement for mirrored axis data.')

    current = FloatItem('Tunnel current', unit='A', default=0,
            help='Tunnel current for the measured data, or start condition\
            for spectra.')
    vgap = FloatItem('Gap voltage', unit='V', default=0,
            help='Gap voltage for the measured data, or start condition for\
            spectra.')

    unit = StringItem("Data physical unit", default='')

    creation_date = StringItem("Measurement data", default='')

    comment = TextItem('Comment',default='')

    processing = TextItem('Processing',default='',
            help='List all processing steps that occured on this data.')

    t = BeginTabGroup("Axis")
    axis1 = AxisParamItem('First Axis')  # for all type of measurement
    axis2 = AxisParamItem('Second Axis') # for topography and 3D map
    axis3 = AxisParamItem('Third Axis')  # for 3D map only
    _t = EndTabGroup("Axis")

    metadata = DictItem("Metadata", default={})

    def copy(self):
        """Return a copy of the object"""
        return deepcopy(self)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError, _("They are no such attributes %s.") % key

    def __setitem__(self, key, value):
        setattr(self, key, value)


# Register the new class as a DataSetItem
class ScanningParamWidget(DataSetWidget):
    klass = ScanningParam
class ScanningParamItem(ObjectItem):
    klass = ScanningParam
DataSetEditLayout.register(ScanningParamItem, ScanningParamWidget)
DataSetShowLayout.register(ScanningParamItem, ScanningParamWidget)


class Measurement(DataSet):
    """Measurement"""

    """"
       This object store the data and physical informations
       about the scanning probe measurement. It also has function to process
       the data.

       Data are stored in the data member in a numpy matrix.
       Info are stored in a DataSet parameter class derived from guidata.
    """

    rawdata = FloatArrayItem(_("Raw measured data"), default=ndarray(0),
                             format=" %.2e ")
    param = ScanningParamItem(_('Scanning parameters'))

    def __init__(self, rawdata=None, param=None,
                 title=None, comment=None, icon=''):
        """\arg rawdata a numpy.ndarray
           \arg param a ScanningParam object
        """
        DataSet.__init__(self, title=title, comment=comment, icon=icon)
        if rawdata is not None:
            self.set_rawdata(rawdata)
        if param is not None :
            self.set_param(param)

    def copy(self, source):
        """Copy a the data and the parameter from a Measurement object."""
        if isinstance(source, Measurement):
            self.copy_param(source.param)
            self.copy_rawdata(source.rawdata)

    def get_rawdata(self):
        """Return the raw data measured in physical, unit according to
        parameter"""
        return self.rawdata

    def set_rawdata(self, source):
        """Set the data from a source (without copying)
        \arg source can be either a Measurement object or data.
        """
        if isinstance(source, Measurement):
            source = source.rawdata
        assert isinstance(source, ndarray), _('Wrong data type')
        self.rawdata = source

    def set_param(self, source):
        """Set the parameter (without copying) from a source.
        \arg source can be either a Measurement object or ScanningParam.
        """
        if isinstance(source, Measurement):
            source = source.param
        assert isinstance(source, ScanningParam), _('Wrong parameter type')
        self.param = source

    def copy_param(self, source):
        """Copy the parameter from a source.
        \arg source can be either a Measurement object or ScanningParam.
        """
        if isinstance(source, Measurement):
            source = source.param
        self.set_param(deepcopy(source))

    def copy_rawdata(self, source):
        """Copy the data from a source.
        \arg source can be either a Measurement object or data.
        """
        if isinstance(source, Measurement):
            source = source.rawdata
        self.set_rawdata(array(source, copy=True))

    def get_dimension(self):
        """Return the dimension of the data"""
        return len(self.rawdata.shape)

    def get_shape(self):
        return self.rawdata.shape

    def get_scale(self):
        """Return the scale as a list along the axis corresponding to the
        type of the measurement."""
        dimension = self.get_dimension()

        if 1 == dimension:
            return self.param.axis1.get_scale()
        elif 2 == dimension:
            return self.param.axis2.get_scale()
        elif 3 == dimension:
            return self.param.axis3.get_scale()

    def __add__(self, a):
        new_param = self.param.copy()
        if isinstance(a, (int,float)):
            new_param.processing += _('add %2.3g') % (a,) + '\n'
            new_data = self.rawdata + a
        elif isinstance(a, Measurement):
            if self.rawdata.shape != a.rawdata.shape:
                raise ValueError, _('Can only add measurement with same shape')
            new_param.processing += _('%s added') % (a.param.filename,) + '\n'
            new_data = self.rawdata + a.rawdata
        else:
            raise TypeError, _('Unsupported operand type')

        return Measurement(new_data, new_param)

    def __mul__(self, a):
        if not isinstance(a, (int,float)):
            raise TypeError, _('Unsupported operand type')
        new_param = self.param.copy()
        new_param.processing += _('scaled by %2.3g') % (a,) + '\n'
        new_data = self.rawdata * a
        return Measurement(new_data, new_param)

    def __div__(self, a):
        if not isinstance(a, (int,float)):
            raise TypeError, _('Unsupported operand type')
        new_param = self.param.copy()
        new_param.processing += _('divided by %2.3g') % (a,) + '\n'
        new_data = self.rawdata / a
        return Measurement(new_data, new_param)

    def __truediv__(self, a):
        if not isinstance(a, (int,float)):
            raise TypeError, _('Unsupported operand type')
        new_param = self.param.copy()
        new_param.processing += _('divided by %2.3g') % (a,) + '\n'
        new_data = self.rawdata / a
        return Measurement(new_data, new_param)

    def __floordiv__(self, a):
        if not isinstance(a, (int,float)):
            raise TypeError, _('Unsupported operand type')
        new_param = self.param.copy()
        new_param.processing += _('divided by %2.3g') % (a,) + '\n'
        new_data = self.rawdata // a
        return Measurement(new_data, new_param)

    def __getitem__(self, key):
        if key == 'rawdata':
            return self.get_rawdata()
        elif key == 'shape':
            return self.get_shape()
        elif key == 'scale':
            return self.get_scale()
        else:
            try:
                return getattr(self.param, key)
            except AttributeError:
                raise KeyError, _("They are no such attributes %s.") % key

    def __setitem__(self, key, value):
        if key == 'rawdata':
            self.set_rawdata(value)
        else:
            setattr(self.param, key, value)

    def __len__(self):
        if self.get_dimension() in (1, 3):
            return self.get_shape()[-1]
        else:
            x, y = self.get_shape()
            return x * y


class ComputationError(Exception):
    pass

class OperationProxy():
    """This class allow to perform computation directly on curve, image items
       like if they were numpy.ndarray. All the performed computation are stored
       in the parameter processing of the item. Any operation return a new item,
       it avoid overwritting raw data.
    """
    QProxy = QObject()

    def __init__(self, parent=None):
        self.__baseclass__ = None # undefined

    def __add__(self, a):
        if isinstance(a, self.__baseclass__) :
            return self.__baseclass__(self.measurement + a.measurement)
        elif isinstance(a, (int, float)):
            return self.__baseclass__(self.measurement + a)
        else:
            raise TypeError, _('Unsupported operand type')

    def __mul__(self, a):
        return self.__baseclass__(self.measurement * a)

    def __div__(self, a):
        return self.__baseclass__(self.measurement / a)

    def __truediv__(self, a):
        return self.__baseclass__(self.measurement / a)

    def __floordiv__(self, a):
        return self.__baseclass__(self.measurement // a)

    def __getitem__(self, key):
        return self.measurement[key]

    def __setitem__(self, key, value):
        self.measurement[key] = value

    def __len__(self):
        return len(self.measurement)

    def get_rawdata(self):
        return self.measurement['rawdata']

    def get_scale(self):
        return self.measurement['scale']

    def get_shape(self):
        return self.measurement['shape']

    def apply_func(self, func, param=None):
        if param is None:
            return func(self.get_rawdata())
        else:
            return func(self.get_rawdata(), param)

    def start_compute(self, name, func, param=None, suffix=None,
                      interactive=True):
        self.QProxy.emit(SIGNAL("computing"))
        if interactive and param is not None:
            if interactive == 'text' and not param.text_edit():
                return
            elif not param.edit():
                return
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:
            processing = name
            if suffix is not None:
                processing += " | " + suffix(param)

            result = Measurement()
            result.copy_param(self.measurement)
            result.set_rawdata(self.apply_func(func, param))
            result['processing'] += processing + "\n"
        except Exception, msg:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(None, APP_NAME,
                                 _(u"An error occured:")+"\n%s" % unicode(msg))
            raise ComputationError, msg
        finally:
            self.QProxy.emit(SIGNAL("computing finished"))
            QApplication.restoreOverrideCursor()
        return self.__baseclass__(result)

    def end_compute(self, result):
        result.update_style()
        self.QProxy.emit(SIGNAL("new_item_computed"), result)
        return result

    def compute(self, name, func, param=None, suffix=None, interactive=True):
        try:
            result = self.start_compute(name, func, param, suffix, interactive)
            return self.end_compute(result)
        except ComputationError, msg:
            print _("Computation error: %s") % msg
            return


class ThothCurveItem(CurveItem, OperationProxy):
    """A derived class from CurveItem which store measurement parameters,
       and create style depending on the curve type.
    """

    def __init__(self, measurement=None):
        """\arg measurement a Measurement object storing data and scanning
        parameter.
        """
        self.__baseclass__ = ThothCurveItem
        self.measurement = Measurement()
        param = CurveParam(_("Line"), icon='curve.png')
        param.line.style = "SolidLine"
        param.line.width = 2.0
        CurveItem.__init__(self, param)
        if measurement is not None:
            self.set_measurement(measurement)

    def update_style(self):
        t = self['type']
        if 'ivcurve' == t:
            self.curveparam.line.color = 'green'
            self.curveparam.title = _('IV curve')
        elif 'didvcurve'== t:
            self.curveparam.line.color = 'green'
            self.curveparam.title = _('dIdV curve')
        else:
            self.curveparam.line.color = 'red'
            self.curveparam.title = _('Unknown curve')
        self.curveparam.update_curve(self)

    def set_measurement(self, measurement):
        """Set the measurement to the curve item"""
        assert isinstance(measurement, Measurement), _(
                'Measurement as not the right type.')
        assert measurement.get_dimension() == 1, _(
                'Dimension is wrong for a curve')
        ## Maybe we could check for the right shape here. (unused)
        #assert measurement.param.axis1.get_length() == len(measurement.data), _(
                #'Data shape does not match axis length.')
        self.measurement = measurement
        self.set_data(x=self.measurement.get_scale(),
                      y=self.measurement.rawdata)
        self.update_style()

    def get_item_parameters(self, itemparams):
        """Return the item parameters, required for the GUI display of the
        measurement parameters."""
        CurveItem.get_item_parameters(self, itemparams)
        itemparams.add("MeasurementParameter", self, self.measurement)

    def set_item_parameters(self, itemparams):
        """Set the item parameters, required for the GUI display of the
        measurement parameters."""
        update_dataset(self.measurement,
                       itemparams.get("MeasurementParameter"),
                       visible_only=True)
        CurveItem.set_item_parameters(self, itemparams)

    def compute_detrend(self):
        """Remove linear trend along axis from data."""
        import scipy.signal as sps
        return self.compute(_('Linear background substracted'), sps.detrend)

    def compute_scale(self, param=None, interactive=True):
        """Multiply the curve by the given factor"""
        class ScaleParam(DataSet):
            factor = FloatItem(_("Scaling factor"), default=1)
        if param is None:
            param = ScaleParam(_("Scaling"))
        return self.compute(_("Scaled"),
            lambda y, p: y * p.factor,
            param,
            lambda p: _("factor = %.3f") % p.factor,
            interactive)

    def compute_shift(self, param=None, interactive=True):
        """Shift the curve by the given amount"""
        class ShiftParam(DataSet):
            shift = FloatItem(_("Shift by"), default=1)
        if param is None:
            param = ShiftParam(_("Shift"))
        return self.compute(_("Shifted"),
            lambda y, p: y + p.shift,
            param,
            lambda p: _("shift = %.3f") % p.shift,
            interactive)

    def compute_savitzky(self, param=None, interactive=True):
        """Smooth or derivate data based on the Savitzky-Golay algorithm"""
        import sgfilter
        class SGParam(DataSet):
            num_points = IntItem(_("Number of points"), default=8, min=2)
            poly_degree = IntItem(_("Polynom degree"), default=4, min=1)
            diff_order = IntItem(_("Differential order"), default=0, min=0)
        if param is None:
            param = SGParam(_("Savitzky-Golay filter"))
        def func(y, p):
            return sgfilter.savitzky(y, p.num_points, p.poly_degree,
                                     p.diff_order)
        try:
            result = self.start_compute(_('Smoothed with Savitzky-Golay'),
                                func,
                                param,
                                lambda p: _("""%i points, polynom degree %i, differential order %i""") % \
                                            (p.poly_degree, p.num_points,
                                            p.diff_order),
                                interactive)
            if param.diff_order == 1 and result['type'] == 'ivcurve':
                    result['type'] = 'didvcurve'
                    result['unit'] = _('a.u.')
            else:
                result['type'] == 'unknowncurve'
            return self.end_compute(result)
        except ComputationError, msg:
            print msg

    def compute_derivative(self):
        """Compute point derivation of the curve"""
        increment = self['axis1']['increment']
        result = self.start_compute(_('Point derivation'),
                              lambda x: (x[1:] - x[:-1]) / increment)
        result['axis1'].length = len(result) - 1
        result['axis1'].start += increment / 2.
        if result['type'] == 'ivcurve':
            result['type'] = 'didvcurve'
            result['unit'] = _('a.u.')
        else:
            result['type'] == 'unknowncurve'
        return self.end_compute(result)

    def compute_spline_derivative(self, param=None, interactive=True):
        """ Compute the derivative of the curve basde on spline
            interpolation"""
        from scipy.interpolate import splrep, splev
        class SplineParam(DataSet):
            s = FloatItem(_("s"), default=0.1, min=0,
                            help=_('''Larger s means more smoothing while
smaller values of s indicate less smoothing. 0 is no smoothing.'''))
        if param is None:
            param = SplineParam(_("Spline smoothing"))
        def func(y, p):
            tck = splrep(arange(len(y)),y,s=p.s) # get spline coef tuple
            return splev(arange(len(y)),tck,der=1) # compute 1st derivative
        result = self.start_compute(_("Spline derivative"), func, param,
                            lambda p: u"s=%i" % p.s,
                            interactive)
        if result['type'] == 'ivcurve':
            result['type'] = 'didvcurve'
            result['unit'] = _('a.u.')
        else:
            result['type'] == 'unknowncurve'
        return self.end_compute(result)

    def compute_fft(self):
        """Compute FFT"""
        import numpy.fft
        from numpy import absolute
        scale = self.get_scale()
        def func(data):
            return absolute(numpy.fft.fft(data))
        result = self.start_compute(_('Fourrier transform'), func)
        new_scale = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       scale.shape[-1], d=scale[1] - scale[0]))
        result['unit'] = _('a.u.')
        result['axis1']['start'] = new_scale[0]
        result['axis1']['increment'] = new_scale[1] - new_scale[0]
        result['axis1']['unit'] = '1/%s' % result['axis1']['unit']
        result['type'] += 'fft'
        return self.end_compute(result)

    def compute_ifft(self):
        """Compute iFFT"""
        import numpy.fft
        from numpy import absolute
        scale = self.get_scale()
        def func(data):
            return absolute(numpy.fft.ifft(data))
        result = self.start_compute(_('Fourrier transform'), func)
        new_scale = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       scale.shape[-1], d=scale[1] - scale[0]))
        result['unit'] = _('a.u.')
        result['axis1']['start'] = new_scale[0]
        result['axis1']['increment'] = new_scale[1] - new_scale[0]
        result['axis1']['unit'] = '1/%s' % result['axis1']['unit']
        return self.end_compute(result)

    def compute_wiener(self):
        """Smooth data with wiener filter"""
        import scipy.signal as sps
        return self.compute(_('Wiener filter'), sps.wiener)

    def compute_gaussian(self, param=None, interactive=True):
        class GaussianParam(DataSet):
            sigma = FloatItem(u"σ", default=1.)
        if param is None:
            param = GaussianParam(_("Gaussian filter"))
        import scipy.ndimage as spi
        def func(y, p):
            return spi.gaussian_filter1d(y, p.sigma)
        result = self.start_compute(_("Gaussian filter"), func, param,
                            lambda p: u"σ=%.3f pixels" % p.sigma,
                            interactive)
        return self.end_compute(result)

    def compute_zeroes(self) :
        """Set the minimum of the data to zero."""
        return self.compute(_('Zeroes'), lambda x: x - x.min())

    def fit_bcs(self):
        from guiqwt.fit import FitParam, FitDialog
        import bcs
        from numpy import sum
        def fit(x, param):
            return bcs.BCS(param[1], param[0], x)

        T = FitParam("Temperature [K]", 2, 0.1, 20)
        gap = FitParam("Gap [meV]", 1.28, 1, 10) # in meV, value for NbSe_2
        params = [T,gap]

        x, y = self.get_data()
        # FIXME : not a very good way to proceed
        # Normalize with 20 point from the left of the curve
        normalisation_range = y[:20]
        normalisation = sum(normalisation_range) / len(normalisation_range)
        y = y / normalisation
        #values = guifit(x, y, fit, params, auto_fit=True)
        win = FitDialog(edit=True, wintitle='BCS fitting', toolbar=True,
                param_cols=1, auto_fit=True,
                options=dict(title="BCS", xlabel="Energy", ylabel="DOS"))
        win.set_data(x, y, fit, params, fitargs=None, fitkwargs=None)
        self.QProxy.emit(SIGNAL("new_window"), win)
        win.autofit()
        win.show()

        return win

class ThothImageItem(ImageItem, OperationProxy):
    """A class derived from ImageItem which store measurement parameters,
       and create style depending on the image type.
    """
    def __init__(self, measurement=None):
        """\arg measurement a Measurement object storing data and scanning
        parameter.
        """
        self.__baseclass__ = ThothImageItem
        param = ImageParam(_("Image"), icon='image.png')
        param.interpolation = 1
        ImageItem.__init__(self, data=None, param=param)
        if measurement is not None:
            self.set_measurement(measurement)
        else:
            self.measurement = Measurement()

    # fix bug in guiqwt
    def get_data(self, x0, y0, x1=None, y1=None):
        """
        Return image data
        Arguments:
          x0, y0 [, x1, y1]
        Return image level at coordinates (x0,y0)
        If x1,y1 are specified:
          return image levels (np.ndarray) in rectangular area (x0,y0,x1,y1)
        """
        if x1 is None or y1 is None:
            i0, j0 = self.get_closest_pixel_indexes(x0, y0)
            return self.data[j0, i0]
        else:
            i0, j0, i1, j1 = self.get_closest_index_rect(x0, y0, x1, y1)
            return (self.get_x_values(i0, i1), self.get_y_values(j0, j1),
                    self.data[j0:j1, i0:i1])

    def update_style(self):
        t = self['type']
        if 'topo' == t:
            self.imageparam.label = _('Topography')
            self.imageparam.colormap = 'gist_heat'
        elif 'topofft'== t:
            self.imageparam.label = _('FFT of topography')
            self.imageparam.colormap = 'jet'
        else:
            self.imageparam.label = _('Unknow type')
            self.imageparam.colormap = 'Reds'
        self.imageparam.update_image(self)

    def set_measurement(self, measurement):
        """Set the measurement to the image item"""
        assert isinstance(measurement, Measurement), _(
                'Measurement as not the right type.')
        assert measurement.get_dimension() == 2, _(
                'Dimension is wrong for a curve')
        ## Maybe we could check for the right shape here. (unused)
        # xsize, ysize = measurement.rawdata.shape
        # assert measurement.param.axis1.get_length() == xsize and
                #measurement.param.axis2.get_length() == ysize, _(
                #'Data shape does not match axis length.')
        self.measurement = measurement
        self.imageparam.xmin = 0
        self.imageparam.ymin = 0
        self.imageparam.xmax = measurement['axis1'].get_physical_length()
        self.imageparam.ymax = measurement['axis2'].get_physical_length()
        self.set_data(measurement.rawdata)
        self.update_style()

    def get_item_parameters(self, itemparams):
        """Return the item parameters, required for the GUI display of the
        measurement parameters."""
        ImageItem.get_item_parameters(self, itemparams)
        itemparams.add("MeasurementParameter", self, self.measurement)

    def set_item_parameters(self, itemparams):
        """Set the item parameters, required for the GUI display of the
        measurement parameters."""
        update_dataset(self.measurement,
                       itemparams.get("MeasurementParameter"),
                       visible_only=True)
        ImageItem.set_item_parameters(self, itemparams)

    def compute_plane_correction(self):
        """Substract least square fitted plane through the image"""
        pass

    def compute_line_detrend(self):
        """Apply a linewise slope correction."""
        import scipy.signal as sps
        def func(data):
            return array([sps.detrend(line) for line in data])
        return self.compute(_('Horizontal line correction'), func)

    def compute_savitzky(self, param=None, interactive=True):
        """Linewise smoothing of data based on the Savitzky-Golay
           algorithm"""
        import sgfilter
        class SGParam(DataSet):
            num_points = IntItem(_("Number of points"), default=8, min=2)
            poly_degree = IntItem(_("Polynom degree"), default=4, min=1)
            diff_order = 0
        if param is None:
            param = SGParam(_("Savitzky-Golay filter"))
        def func(data, p):
            return array([sgfilter.savitzky(line, p.num_points, p.poly_degree,
                                     p.diff_order) for line in data])
        return self.compute(_('Smoothed with Savitzky-Golay'),
                            func,
                            param,
                            lambda p: _("""%i points, polynom degree %i, differential order %i""") % \
                                        (p.poly_degree, p.num_points,
                                         p.diff_order),
                            interactive=interactive)

    def compute_fft(self):
        """Compute FFT"""
        import numpy.fft
        from numpy import absolute
        def func(data):
            return numpy.fft.fftshift(absolute(numpy.fft.fft2(data)))
        result = self.start_compute(_('Fourrier transform'), func)
        new_x = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       self['axis1']['length'], d=self['axis1']['increment']))
        new_y = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       self['axis2']['length'], d=self['axis2']['increment']))
        result['unit'] = _('a.u.')
        result['axis1']['start'] = new_x[0]
        result['axis1']['increment'] = new_x[1] - new_x[0]
        result['axis1']['unit'] = '1/%s' % result['axis1']['unit']
        result['axis2']['start'] = new_y[0]
        result['axis2']['increment'] = new_y[1] - new_y[0]
        result['axis2']['unit'] = '1/%s' % result['axis2']['unit']
        result['type'] += 'fft'
        return self.end_compute(result)

    def compute_ifft(self):
        """Compute iFFT"""
        import numpy.fft
        from numpy import absolute
        def func(data):
            return numpy.fft.fftshift(absolute(numpy.fft.ifft2(data)))
        result = self.start_compute(_('Fourrier transform'), func)
        new_x = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       self['axis1']['length'],
                                       d=self['axis1']['increment']))
        new_y = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       self['axis2']['length'],
                                       d=self['axis2']['increment']))
        result['unit'] = _('a.u.')
        result['axis1']['start'] = new_x[0]
        result['axis1']['increment'] = new_x[1] - new_x[0]
        result['axis1']['unit'] = '1/%s' % result['axis1']['unit']
        result['axis2']['start'] = new_y[0]
        result['axis2']['increment'] = new_y[1] - new_y[0]
        result['axis2']['unit'] = '1/%s' % result['axis2']['unit']
        return self.end_compute(result)

    def compute_wiener(self):
        """Smooth data with wiener filter"""
        import scipy.signal as sps
        return self.compute(_('Wiener filter'), sps.wiener)

    def compute_gaussian(self, param=None, interactive=True):
        class GaussianParam(DataSet):
            sigma = FloatItem(u"σ", default=1.)
        if param is None:
            param = GaussianParam(_("Gaussian filter"))
        import scipy.ndimage as spi
        def func(y, p):
            return spi.gaussian_filter(y, p.sigma)
        return self.compute(_("Gaussian filter"), func, param,
                            suffix=lambda p: u"σ=%.3f pixels" % p.sigma,
                            interactive=interactive)

    def compute_zeroes( self ) :
        """Set the minimum of the data to zero."""
        return self.compute(_('Zeroes'), lambda x: x - x.min())


class ThothMapItem(ImageItem, OperationProxy):
    """A class derived from ImageItem which store the 3D map a list of images
    the measurement parameters, and create style depending on the image type.
    The class also implement the capability to change the current slice, get the
    curve (across the image list) at a given pixel or the averaged curve on a
    region.
    """
    def __init__(self, measurement=None):
        """\arg measurement a Measurement object storing data and scanning
        parameter.
        """
        self.__baseclass__ = ThothMapItem
        ImageItem.__init__(self, data=None, param=ImageParam(_("Image")))
        if measurement is not None:
            self.set_measurement(measurement)
        else:
            self.measurement = Measurement()
        self.current_index = 0

    # fix bug in guiqwt
    def get_data(self, x0, y0, x1=None, y1=None):
        """
        Return image data
        Arguments:
          x0, y0 [, x1, y1]
        Return image level at coordinates (x0,y0)
        If x1,y1 are specified:
          return image levels (np.ndarray) in rectangular area (x0,y0,x1,y1)
        """
        if x1 is None or y1 is None:
            i0, j0 = self.get_closest_pixel_indexes(x0, y0)
            return self.data[j0, i0]
        else:
            i0, j0, i1, j1 = self.get_closest_index_rect(x0, y0, x1, y1)
            return (self.get_x_values(i0, i1), self.get_y_values(j0, j1),
                    self.data[j0:j1, i0:i1])

    def update_style(self):
        t = self['type']
        if 'ivmap' == t:
            self.imageparam.label = _('IV map')
            self.imageparam.colormap = 'Blues'
        elif 'didvmap' == t:
            self.imageparam.label = _('dIdV map')
            self.imageparam.colormap = 'Purples'
        elif 'ivmapfft' == t:
            self.imageparam.label = _('FFT of IV map')
            self.imageparam.colormap = 'jet'
        elif 'didvmapfft' == t:
            self.imageparam.label = _('FFT of dIdV map')
            self.imageparam.colormap = 'jet'
        else:
            self.imageparam.label = _('Unknow type')
            self.imageparam.colormap = 'Reds'
        self.imageparam.update_image(self)

    def set_measurement(self, measurement):
        """Set the measurement to the map item"""
        assert isinstance(measurement, Measurement), _(
                'Measurement as not the right type.')
        assert measurement.get_dimension() == 3, _(
                'Dimension is wrong for a 3D map')
        ## Maybe we could check for the right shape here. (unused)
        # xsize, ysize, zsize = measurement.rawdata.shape
        # assert measurement.param.axis1.get_length() == xsize and
                #measurement.param.axis2.get_length() == ysize,
                #measurement.param.axis3.get_length() == ysize,_(
                #'Data shape does not match axis length.')
        self.measurement = measurement
        self.set_data(measurement.rawdata[0,:,:])
        self.imageparam.xmin = 0
        self.imageparam.ymin = 0
        self.imageparam.xmax = measurement['axis1'].get_physical_length()
        self.imageparam.ymax = measurement['axis2'].get_physical_length()
        self.update_style()

    def get_item_parameters(self, itemparams):
        """Return the item parameters, required for the GUI display of the
        measurement parameters."""
        ImageItem.get_item_parameters(self, itemparams)
        itemparams.add("MeasurementParameter", self, self.measurement)

    def set_item_parameters(self, itemparams):
        """Set the item parameters, required for the GUI display of the
        measurement parameters."""
        update_dataset(self.measurement,
                       itemparams.get("MeasurementParameter"),
                       visible_only=True)
        ImageItem.set_item_parameters(self, itemparams)

    def get_current_index(self):
        """Return the current slice index display/stored in the image"""
        return self.current_index

    def set_slice_index(self, index):
        """Set the current slice to index
            \arg index an integer like for python list index
        """
        assert isinstance(index, int), _('Slice index has to be an integer')
        index_max = self.measurement.param.axis3.get_length()
        if abs(index) > index_max:
            raise IndexError, _('Slice index out of bounds')

        self.set_data(self.measurement.rawdata[index,:,:])
        if index < 0:
            self.current_index = index_max + index # index is negative
        else:
            self.current_index = index

    def set_slice_value(self, value):
        """Set the current slice to a given value. If the value is not in the
        measured discrete scale, it will interpolate data between the adjacent
        slices.
            \arg value a float within scale.min() and scale.max()
        """
        assert isinstance(value, (int, float)), _(
                                'Slice value has to be a float')
        scale = self.measurement.param.axis3.get_scale()
        if not scale.min() <= value <= scale.max():
            raise ValueError, _('Slice value out of range')

        if value in scale:
            # get the index with value nearest to v
            index = where(scale == value)[0][0]
            self.set_data(self.measurement.rawdata[index,:,:])
        else:
            # get the lowest index with value nearest to v
            index = where(scale > value )[0][0]
            coef = (value - scale[index]) / (scale[index+1] - scale[index])
            rawdata = self.measurement.rawdata
            interpolated_data = rawdata[index,:,:] * (1 - coef) \
                    + rawdata[index+1,:,:] * coef
            self.set_data(interpolated_data)
        self.current_index = index

    def get_curve_at_position(self, x0, y0, x1=None, y1=None):
        if x1 is None and y1 is None:
            return self.get_curve_at_index(
                                *self.get_closest_pixel_indexes(x0, y0))
        else:
            i0, j0 = self.get_closest_pixel_indexes(x0, y0)
            i1, j1 = self.get_closest_pixel_indexes(x1, y1)
            return self.get_curve_at_index(i0, j0, i1, j1)

    def get_curve_at_index(self, i0, j0, i1=None, j1=None):
        """
        Return the curve across the slices at the give pixel coordinates.
        If two points are specified, the averaged curve is returned within
        the selected area.
        """
        maxx, maxy, maxz = self.measurement.rawdata.shape
        if not ( 0 <= i0 < maxx and 0 <= j0 < maxy ):
            raise IndexError, _('Out of bounds')
        if i1 is not None and j1 is None:
            raise TypeError, _('Missing argument')

        # FIXME set it as a standard compute function
        #result = self.start_compute('Extract curve ', lambda x, p: x[:, j0, i0],
        #param=
        #suffix=lambda

        new_param = self.measurement.param.copy()
        if new_param.type == 'didvmap':
            new_param.type = 'didvcurve'
        elif new_param.type == 'ivmap':
            new_param.type = 'ivcurve'
        new_param.axis1 = new_param.axis3
        new_param.axis2 = AxisParam()
        new_param.axis3 = AxisParam()

        if i1 is None : # only one pixel
            x0, y0 = self.get_plot_coordinates(i0, j0)
            xunit = self['axis1']['unit']
            yunit = self['axis2']['unit']
            new_param.processing += \
                            _('Extracted from (%2.3g %s , %2.3g %s) \n') % \
                                      (x0, xunit, y0, yunit)
            new_data = self.measurement.rawdata[:, j0, i0]
            new_measurement = Measurement(new_data, new_param)
            return ThothCurveItem(new_measurement)
        else: # average over many pixels
            if not ( 0 <= i1 < maxx and 0 <= j1 < maxy ):
                raise IndexError, _('Out of bounds')
            #new_param.processing += _(
                        #'average extracted from (%2.3g,%2.3g,%2.3g,%2.3g,)') % \
                        #(self.get_plot_coordinates(i0, j0),
                         #self.get_plot_coordinates(i1, j1))
            j1 += 1
            i1 += 1
            new_data = self.measurement.rawdata[:, j0:j1, i0:i1].sum(1).sum(1) \
                            / (max(abs(i1 - i0), 1) * (max(abs(j1 - j0), 1)))
                       # average curve by summing matrix
                       # along second axis twice + div
            new_measurement = Measurement(new_data, new_param)
            return ThothCurveItem(new_measurement)

    def compute_get_curve_at_index(self, i0, j0, i1=None, j1=None):
        """Identical to get_curve_at but send Signal to Thoth manager"""
        result = self.get_curve_at_index(i0, j0, i1, j1)
        return self.end_compute(result)

    def compute_get_curve_at_position(self, x0, y0, x1=None, y1=None):
        """Identical to get_curve_at but send Signal to Thoth manager"""
        result = self.get_curve_at_position(x0, y0, x1, y1)
        return self.end_compute(result)

    def compute_savitzky(self, param=None, interactive=True):
        """Spectrum wise smoothing or derivate data based on the Savitzky-Golay
           algorithm"""
        import sgfilter
        class SGParam(DataSet):
            num_points = IntItem(_("Number of points"), default=8, min=2)
            poly_degree = IntItem(_("Polynom degree"), default=4, min=1)
            diff_order = IntItem(_("Differential order"), default=0, min=0)
        if param is None:
            param = SGParam(_("Savitzky-Golay filter"))
        def func(data, p):
            slices, rows, cols = data.shape
            # create a view where each line represents one spectra
            spectra = data.reshape(slices, rows*cols).transpose()
            result = array([sgfilter.savitzky(s, p.num_points, p.poly_degree,
                                              p.diff_order) for s in spectra])
            return result.transpose().reshape(slices, rows, cols)
        result = self.start_compute(_('Smoothed with Savitzky-Golay'),
                            func,
                            param,
                            lambda p: _("%i points, polynom degree %i, differential order %i") % \
                                        (p.poly_degree, p.num_points,
                                         p.diff_order),
                            interactive=interactive)
        if param.diff_order == 0:
            return result
        elif param.diff_order == 1 and result['type'] == 'ivmap':
                result['type'] = 'didvmap'
                result['unit'] = _('a.u.')
        else:
            result['type'] == 'unknownmap'
        return self.end_compute(result)

    def compute_fft(self):
        """Compute FFT"""
        import numpy.fft
        from numpy import absolute
        def func(data):
            return array([numpy.fft.fftshift(absolute(numpy.fft.fft2(slice_)))
                                                          for slice_ in data])
        result = self.start_compute(_('Fourrier transform'), func)
        new_x = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       self['axis1']['length'], d=self['axis1']['increment']))
        new_y = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       self['axis2']['length'], d=self['axis2']['increment']))
        result['unit'] = _('a.u.')
        result['axis1']['start'] = new_x[0]
        result['axis1']['increment'] = new_x[1] - new_x[0]
        result['axis1']['unit'] = '1/%s' % result['axis1']['unit']
        result['axis2']['start'] = new_y[0]
        result['axis2']['increment'] = new_y[1] - new_y[0]
        result['axis2']['unit'] = '1/%s' % result['axis2']['unit']
        result['type'] += 'fft'
        return self.end_compute(result)

    def compute_ifft(self):
        """Compute iFFT"""
        import numpy.fft
        from numpy import absolute
        def func(data):
            return array([numpy.fft.fftshift(absolute(numpy.fft.ifft2(slice_)))
                                                           for slice_ in data])
        result = self.start_compute(_('Fourrier transform'), func)
        new_x = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       self['axis1']['length'], d=self['axis1']['increment']))
        new_y = numpy.fft.fftshift(numpy.fft.fftfreq(
                                       self['axis2']['length'], d=self['axis2']['increment']))
        result['unit'] = _('a.u.')
        result['axis1']['start'] = new_x[0]
        result['axis1']['increment'] = new_x[1] - new_x[0]
        result['axis1']['unit'] = '1/%s' % result['axis1']['unit']
        result['axis2']['start'] = new_y[0]
        result['axis2']['increment'] = new_y[1] - new_y[0]
        result['axis2']['unit'] = '1/%s' % result['axis2']['unit']
        return self.end_compute(result)

    def compute_wiener(self):
        """Smooth data with wiener filter"""
        import scipy.signal as sps
        def func(data):
            return array([sps.wiener(slice_) for slice_ in data])
        return self.compute(_('Wiener filter'), func)

    def compute_gaussian(self, param=None, interactive=True):
        class GaussianParam(DataSet):
            sigma = FloatItem(u"σ", default=1.)
        if param is None:
            param = GaussianParam(_("Gaussian filter"))
        import scipy.ndimage as spi
        def func(data, p):
            return array([spi.gaussian_filter(slice_, p.sigma)
                                                    for slice_ in data])
        return self.compute(_("Gaussian filter"), func, param,
                            suffix=lambda p: u"σ=%.3f pixels" % p.sigma,
                            interactive=interactive)

if __name__ == "__main__":
    pass