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

from flatfile import FlatFile
from thotharray import Measurement, ScanningParam, AxisParam

from guiqwt.config import _

class FlatFileProxy(object):
    """A proxy to convert basic FlatFile dataArray to a Thoth Measurement"""

    def __init__(self):
        self.measurements = []

    def open(self,filename):
        datalist = FlatFile(filename).getData()

        for dataArray in datalist:
            self.measurements.append(self.convert_to_measurement(dataArray))

    def get_measurements(self):
       return self.measurements

    def convert_to_measurement(self, dataArray):
        data = dataArray.data
        param = ScanningParam()
        param.filename = dataArray.info['filename']
        param.type = dataArray.info['type']
        param.direction = dataArray.info['direction']
        param.current = dataArray.info['current']
        param.vgap = dataArray.info['vgap']
        unit = dataArray.info['unit']
        # For convenience scale units
        if param.type == 'topo' and unit == 'm':
            unit = u'nm'
            data = data * 1e9
        elif param.type == 'ivcurve' and unit == 'A':
            unit = u'nA'
            data = data * 1e9
        elif param.type == 'ivmap' and unit == 'A':
            unit = u'nA'
            data = data * 1e9
        param.unit = unit
        param.creation_date = dataArray.info['date']
        param.comment = dataArray.info['comment']

        axis1 = AxisParam()
        axis2 = AxisParam()
        axis3 = AxisParam()

        if param.type in ('topo','ivmap'):
            axis1.length = dataArray.info['xres']
            axis1.unit = dataArray.info['unitxy']
            #axis1.length_physical = dataArray.info['xreal']
            axis1.increment = dataArray.info['xinc']
            axis1.start = 0

            axis2.length = dataArray.info['yres']
            axis2.unit = dataArray.info['unitxy']
            #axis2.length_physical = dataArray.info['yreal']
            axis2.increment = dataArray.info['yinc']
            axis2.start = 0

            param.axis1 = axis1
            param.axis2 = axis2

        if param.type in ('ivcurve','ivmap'):
            if param.type == 'ivcurve':
                axisv = axis1
                param.axis1 = axisv
            else:
                axisv = axis3
                param.axis3 = axisv
            axisv.unit = dataArray.info['unitv']
            coef = 1
            # For convenience scale units
            if axisv.unit == 'V':
                axisv.unit = 'mV'
                coef = 1e3
            axisv.length = dataArray.info['vres']
            #axisv.length_physical = dataArray.info['vreal'] * coef
            axisv.increment = dataArray.info['vinc'] * coef
            axisv.start = dataArray.info['vstart'] * coef

        return Measurement(data, param)

if __name__ == "__main__":
    pass