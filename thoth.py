# -*- coding: utf-8 -*-

"""

    \mainpage Thoth – Scanning Probe Microscopy (SPM) Analysis Tools

    Thoth is a SPM data analysis tools handling 1d, 2d and 3d data types. It is
    based on Guidata and Guiqwt (Qt based) for the GUI. Thoth is written in
    order to be able to use scripted analysis on large bunch of files as well
    as an interactive GUI mode. Thoth is written in python.

    \section Infos

    Thoth written by François Bianco, University of Geneva - francois.bianco@unige.ch
    based on numpy, guidata, guiqwt

    \section Copyright

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


    \section Updates

    2011    Rewrite to guidata,guiqwt version

"""

import re
from numpy import array, ndarray, sum

from guiqwt.config import _
from guidata.qt.QtCore import Qt, QObject, SIGNAL
from guidata.qt.QtGui import (QFileDialog, QWidget, QDockWidget, QSpinBox,
                              QDoubleSpinBox, QGridLayout, QSlider, QAction)
from guidata.qthelpers import create_action, add_actions
from guiqwt.plot import CurveWindow, ImageWindow
from guiqwt.annotations import (AnnotatedCircle, AnnotatedPoint,
                               AnnotatedRectangle)
from guiqwt.styles import CurveParam, ImageParam
from guiqwt.tools import (RectangleTool, EllipseTool, HRangeTool, PlaceAxesTool,
                          MultiLineTool, FreeFormTool, SegmentTool, CircleTool,
                          AnnotatedRectangleTool, AnnotatedEllipseTool,
                          AnnotatedSegmentTool, AnnotatedCircleTool, LabelTool,
                          AnnotatedPointTool, ImageStatsTool,
                          ImageStatsRectangle,
                          VCursorTool, HCursorTool, XCursorTool,
                          ObliqueRectangleTool, AnnotatedObliqueRectangleTool,
                          SelectPointTool, DefaultToolbarID)
from guiqwt.builder import make

from thotharray import *
from flatfileproxy import FlatFileProxy

class Weight(ImageStatsRectangle):

    def get_infos(self):
        if self.image_item is not None:
            x, y, z = self.image_item.get_data(*self.get_rect())
            result = sum(z) * self.image_item['axis1']['increment'] * \
                              self.image_item['axis2']['increment']
            return _(u"Weight:") + u" %.3g" % result

class WeightTool(ImageStatsTool):
    TITLE = _("Weight")

    def create_shape(self):
        annotation = Weight(0, 0, 1, 1)
        self.set_shape_style(annotation)
        self.action.setEnabled(False)
        return annotation, 0, 2

class ExtractCurveTool(SelectPointTool):
    TITLE = _("Curve extraction")
    ICON = "point_selection.png"
    MARKER_STYLE_SECT = "plot"
    MARKER_STYLE_KEY = "marker/curve"
    CURSOR = Qt.PointingHandCursor

    def __init__(self, manager, mode="create", on_active_item=False,
                 title=None, icon=None, tip=None, end_callback=None,
                 toolbar_id=DefaultToolbarID, marker_style=None):
        SelectPointTool.__init__(self, manager, mode=mode,
                                on_active_item=on_active_item,
                                title=title, icon=icon, tip=tip,
                                end_callback=self.extract_curve,
                                toolbar_id=toolbar_id,
                                marker_style=marker_style)

    def extract_curve(self, x):
        active = self.get_active_plot().get_active_item(force=True)
        if not isinstance(active, ThothMapItem):
            return
        active.compute_get_curve_at_position(*self.get_coordinates())


def create_action(parent, title, triggered=None, toggled=None,
                  shortcut=None, icon=None, tip=None, checkable=None,
                  context=Qt.WindowShortcut, enabled=None):
    """
    Helper to create a new QAction
    """
    action = QAction(title, parent)
    if triggered:
        parent.connect(action, SIGNAL("triggered()"), triggered)
    if checkable is not None:
        # Action may be checkable even if the toggled signal is not connected
        action.setCheckable(checkable)
    if toggled:
        parent.connect(action, SIGNAL("toggled(bool)"), toggled)
        action.setCheckable(True)
    if icon is not None:
        assert isinstance(icon, QIcon)
        action.setIcon( icon )
    if shortcut is not None:
        action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if enabled is not None:
        action.setEnabled(enabled)
    action.setShortcutContext(context)
    return action

class ThothCurveWindow(CurveWindow):
    pass

class MapWindow(ImageWindow):

    def set_map(self, item):
        self.map_ = item

    def wheelEvent(self, event):

        if self.map_ is None:
            return

        # from qt example
        num_degrees = event.delta() / 8.
        num_steps = int(round(num_degrees / 15.))

        if event.orientation() == Qt.Vertical :
            try: # maybe the map has not been set
                new_index = self.map_.get_current_index() + num_steps
                max_index = self.map_.measurement.param.axis3.get_length()
                if new_index > max_index:
                    new_index = max_index
                new_index = int(new_index)
                self.slice_index_spin_box.setValue(new_index)
            except NameError:
                pass

    def set_slice_index(self, index):
        self.map_.set_slice_index(index)
        self.replot()

        scale = self.map_.measurement.param.axis3.get_scale()
        value = scale[index]
        self.slice_value_spin_box.blockSignals(True)
        self.slice_value_spin_box.setValue(value)
        self.slice_value_spin_box.blockSignals(False)

    def set_slice_value(self, value):
        self.map_.set_slice_value(value)
        self.replot()

        index = self.map_.get_current_index()
        self.slice_index_spin_box.blockSignals(True)
        self.slice_index_spin_box.setValue(index)
        self.slice_index_spin_box.blockSignals(False)

    def replot(self):
        plot = self.get_plot()
        plot.replot()
        self.update_cross_sections()
        contrast = self.get_contrast_panel()
        if contrast is not None:
            contrast.histogram.selection_changed(plot)
            #self.set_contrast_range(min, max)

    def create_slice_dock(self):
        widget = QWidget()
        dock = QDockWidget("Slice", self)
        dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea |
                             Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        layout = QGridLayout(widget)
        self.slice_index_spin_box = QSpinBox()
        index_max = self.map_.measurement.param.axis3.get_length()
        self.slice_index_spin_box.setRange(0, index_max-1 )

        slice_slider = QSlider(Qt.Horizontal)
        slice_slider.setRange(0, index_max-1)
        self.connect(self.slice_index_spin_box, SIGNAL("valueChanged(int)"),
                     self.set_slice_index)
        self.connect(slice_slider, SIGNAL("valueChanged(int)"),
                     self.slice_index_spin_box.setValue)
        self.connect(self.slice_index_spin_box, SIGNAL("valueChanged(int)"),
                     slice_slider.setValue)
        self.slice_value_spin_box = QDoubleSpinBox()
        scale = self.map_.measurement.param.axis3.get_scale()
        self.slice_value_spin_box.setRange(scale.min(),scale.max())
        self.slice_value_spin_box.setValue(scale[0])
        self.connect(self.slice_value_spin_box, SIGNAL("valueChanged(double)"),
                     self.set_slice_value)
        unit = self.map_.measurement.param.axis3.unit
        self.slice_value_spin_box.setSuffix(unit)
        layout.addWidget(slice_slider, 0, 0)
        layout.addWidget(self.slice_index_spin_box, 0, 1)
        layout.addWidget(self.slice_value_spin_box, 0, 2)
        dock.setWidget(widget)

class Thoth(QObject):

    ffp = FlatFileProxy()
    opp = OperationProxy()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.connect(self.opp.QProxy, SIGNAL("new_item_computed"),
                self.register_item)
        self.connect(self.opp.QProxy, SIGNAL("new_window"),
                self.register_window)
        self.items = []
        self.windows = []

    def open(self, filename=None, *args):
        if filename is None or isinstance(filename, (int, bool)):
                            # catch Qt signal-slot connection which may
                            # sends boolean from QAction signal
            filename = QFileDialog.getOpenFileNames(None,
                                                    _('Open file'), '.')
            if filename is None:
                return
            else:
                filename = [unicode(fname) for fname in filename]
                self.open(filename)
                return
        elif not isinstance(filename, (list,tuple)):
            filename = [filename,]

        if len(args) != 0:
            filename.extend(args)

        for fname in filename:
            if re.match('.*flat$', fname):
                self.ffp.open(fname)
                measurements = self.ffp.get_measurements()
                for i in range(len(measurements)):
                    self.register_measurement(measurements.pop())
            else:
                print "Unknown file type"

    def get_items(self):
        return self.items

    def get_windows(self):
        return self.windows

    def register_measurement(self, measurements):
        if not isinstance(measurements, (tuple, list)):
            measurements = (measurements,)
        for measurement in measurements:
            type_ = measurement.param.type
            if type_ in ('ivcurve', 'didvcurve'):
                self.register_item(ThothCurveItem(measurement))
            elif type_ in ('topo','topofft'):
                self.register_item(ThothImageItem(measurement))
            elif type_ in ('ivmap','ivmapfft','didvmap','didvmapfft'):
                self.register_item(ThothMapItem(measurement))
            else:
                raise ValueError, _('Not supported measurement type')

    def register_item(self, items):
        if not isinstance(items, (tuple, list)):
            items = (items,)
        for item in items:
            self.items.append(item)
            self.emit(SIGNAL("item_registred"), item)

    def register_window(self, windows):
        if not isinstance(windows, (tuple, list)):
            windows = (windows,)
        for window in windows:
            self.windows.append(window)
            self.emit(SIGNAL("window_registred"), window)
            window.show()

    def create_window(self, items):
        if not isinstance(items, (tuple, list)):
            items = (items,)
        for item in items:
            if isinstance(item, ThothCurveItem):
                window = self._create_curve_window(item)
            elif isinstance(item, ThothImageItem):
                window = self._create_image_window(item)
            elif isinstance(item, ThothMapItem):
                window = self._create_map_window(item)
            else:
                raise ValueError, _('Not supported measurement type')
            other_tools = window.addToolBar(_("Other tools"))
            window.add_toolbar(other_tools,
                               toolbar_id="other_tools")
            for toolklass in (LabelTool, HRangeTool,
                              VCursorTool, HCursorTool, XCursorTool,
                              SegmentTool, RectangleTool, ObliqueRectangleTool,
                              CircleTool, EllipseTool,
                              MultiLineTool, FreeFormTool, PlaceAxesTool,
                              AnnotatedRectangleTool,
                              AnnotatedObliqueRectangleTool,
                              AnnotatedCircleTool, AnnotatedEllipseTool,
                              AnnotatedSegmentTool, AnnotatedPointTool,
                              WeightTool,
                              SelectPointTool,
                              ExtractCurveTool,
                              ):
                window.add_tool(toolklass, toolbar_id="other_tools")

            self.register_window(window)

    def _create_curve_window(self, item):
        window = ThothCurveWindow(wintitle=item.measurement.param.filename,
                             icon="curve.png", toolbar=True)
        compute_menu = window.menuBar().addMenu(_("&Compute"))
        compute_actions = []
        compute_actions.append(create_action(self, _("Scale"),
                          triggered=item.compute_scale,
                          tip=_("Scale the curve.")))
        compute_actions.append(create_action(self,_("Shift"),
                          triggered=item.compute_shift,
                          tip=_("Shift the curve")))
        compute_actions.append(create_action(self, _("&Derivative"),
                          triggered=item.compute_derivative,
                          tip=_("Compute point wise derivative of the curve.")))
        compute_actions.append(create_action(self, _("&Spline derivative"),
                          triggered=item.compute_spline_derivative,
                          tip=_("""Compute the first derivative with spline
interpolation.""")))
        compute_actions.append(create_action(self, _("&Fourier transform"),
                          triggered=item.compute_fft,
                          tip=_("Compute the Fourier transfom of the curve.")))
        compute_actions.append(create_action(self, _("&Inverse Fourier transfom"),
                          triggered=item.compute_ifft,
                          tip=_("Compute the inverse Fourier transfom of the curve.")))
        compute_actions.append(create_action(self, _("&Apply Wiener filter"),
                          triggered=item.compute_wiener,
                          tip=_("Smooth the curve with a Wiener filter.")))
        compute_actions.append(create_action(self, _("&Apply Gauss filter"),
                          triggered=item.compute_gaussian,
                          tip=_("Smooth the curve with a Gaussian filter.")))
        compute_actions.append(create_action(self, _("&Apply Savitzky-Golay"),
                          triggered=item.compute_savitzky,
                          tip=_("Compute a smoothig or a derivative of the curve according to Savitzky-Golay algorithm.")))
        compute_actions.append(create_action(self, _("Fit BCS"),
                          triggered=item.fit_bcs))
        add_actions(compute_menu, compute_actions)
        plot = window.get_plot()
        plot.set_antialiasing(True)
        t = item.measurement.param.type
        if 'ivcurve' == t:
            xlabel = _('Voltage')
            ylabel = _('Current')
        elif 'didvcurve'== t:
            xlabel = _('Energy')
            ylabel = _('Density of state')
        else:
            xlabel = _('Unknown')
            ylabel = _('Unknown')
        xunit = item.measurement.param.axis1.unit
        yunit = item.measurement.param.unit
        plot.set_titles(title=None, xlabel=xlabel, ylabel=ylabel,
                        xunit=xunit, yunit=yunit)
        plot.add_item(item)
        return window

    def _create_image_window(self, item):
        window = ImageWindow(wintitle=item.measurement.param.filename,
                             icon="image.png", toolbar=True)
        compute_menu = window.menuBar().addMenu(_("&Compute"))
        compute_actions = []
        compute_actions.append(create_action(self, _("&Line correction"),
                          triggered=item.compute_line_detrend,
                          tip=_("Apply a linewise slope correction.")))
        compute_actions.append(create_action(self, _("&Fourier transform"),
                          triggered=item.compute_fft,
                          tip=_("Compute the Fourier transfom of the curve.")))
        compute_actions.append(create_action(self, _("&Inverse Fourier transfom"),
                          triggered=item.compute_ifft,
                          tip=_("Compute the inverse Fourier transfom of the curve.")))
        compute_actions.append(create_action(self, _("&Apply Wiener filter"),
                          triggered=item.compute_wiener,
                          tip=_("Smooth the curve with a Wiener filter.")))
        compute_actions.append(create_action(self, _("&Apply Gauss filter"),
                          triggered=item.compute_gaussian,
                          tip=_("Smooth the curve with a Gaussian filter.")))
        compute_actions.append(create_action(self, _("&Apply Savitzky-Golay"),
                          triggered=item.compute_savitzky,
                          tip=_("Compute a smoothig or a derivative of the curve according to Savitzky-Golay algorithm.")))
        compute_actions.append(create_action(self, _("&Zeroes"),
                          triggered=item.compute_zeroes,
                          tip=_("Set the minimum of the data to zero.")))
        add_actions(compute_menu, compute_actions)
        plot = window.get_plot()
        xunit = item.measurement.param.axis1.unit
        yunit = item.measurement.param.axis2.unit
        zunit = item.measurement.param.unit
        plot.set_titles(title=None, xlabel=None, ylabel=None,
                        xunit=xunit, yunit=(yunit,zunit))
        plot.set_axis_direction('left', False)
        plot.add_item(item)
        return window

    def _create_map_window(self, item):
        window = MapWindow(wintitle=item.measurement.param.filename,
                           icon="thoth.png", toolbar=True)
        compute_menu = window.menuBar().addMenu(_("&Compute"))
        compute_actions = []
        compute_actions.append(create_action(self, _("&Fourier transform"),
                          triggered=item.compute_fft,
                          tip=_("Compute the Fourier transfom of the curve.")))
        compute_actions.append(create_action(self, _("&Inverse Fourier transfom"),
                          triggered=item.compute_ifft,
                          tip=_("Compute the inverse Fourier transfom of the curve.")))
        compute_actions.append(create_action(self, _("&Apply Wiener filter"),
                          triggered=item.compute_wiener,
                          tip=_("Smooth the map slicewise with a Wiener filter.")))
        compute_actions.append(create_action(self, _("&Apply Gauss filter"),
                          triggered=item.compute_gaussian,
                          tip=_("Smooth the map slicewise with a Gaussian filter.")))
        compute_actions.append(create_action(self, _("&Apply Savitzky-Golay"),
                          triggered=item.compute_savitzky,
                          tip=_("Compute a smoothig or a derivative spectrum wise according to Savitzky-Golay algorithm.")))
        add_actions(compute_menu, compute_actions)
        plot = window.get_plot()
        xunit = item.measurement.param.axis1.unit
        yunit = item.measurement.param.axis2.unit
        zunit = item.measurement.param.unit
        plot.set_titles(title=None, xlabel=None, ylabel=None,
                        xunit=xunit, yunit=(yunit,zunit))
        plot.set_axis_direction('left', False)
        plot.add_item(item)
        window.set_map(item)
        window.create_slice_dock()
        return window

    def get_average(self, start=0, end=-1):
        sum_ = None
        for item in self.items[start:end]:
            if sum_ is None:
                sum_ = item
            else:
                sum_ += item
        return sum_ / len(self.items[start:end])

def run():
    from guidata import qapplication
    _app = qapplication()
    _app.exec_()

if __name__ == "__main__":
    pass
