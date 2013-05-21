# -*- coding: utf-8 -*-

"""
    Copyright © 2011 François Bianco, University of Geneva -
francois.bianco@unige.ch

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

from nanonisfile import NanonisFile, UnhandledFileError
from thotharray import Measurement, ScanningParam, AxisParam

from guiqwt.config import _

class NanonisFileProxy(object):
    """A proxy to convert Nanonis SXM file to a Thoth Measurement"""

    def __init__(self):
        self.measurements = []

    def open(self,filename):
        datalist = NanonisFile(filename).getData()

        for dataArray in datalist:
            self.measurements.append(self.convert_to_measurement(dataArray))

    def get_measurements(self):
       return self.measurements

    def convert_to_measurement(self, dataArray):
        data = dataArray.data
        param = ScanningParam()
        param.filename = dataArray.info['filename']
        if dataArray.info['data_info'][0]['Name'] != 'Z' :
            raise UnhandledFileError, \
                    'currently only topography is supported.'
        else:
            param.type = 'topo'
        n = dataArray.info['channel']
        if 0 == (n % 2):
            param.direction = 'Forward'
        elif 1 == (n % 2):
            if 'both' == dataArray.info['data_info'][n//2*2-1]['Direction']:
                param.direction = 'Backward'
            else:
                param.direction = 'Forward'

        param.current = float(dataArray.info['z-controller']['Setpoint'].split(\
                                                                       ' ')[0])
        param.vgap = dataArray.info['bias']

        
        # BUG will not work for multichannel files:
        unit = dataArray.info['data_info'][0]['Unit']
        
        # For convenience scale units
        if param.type == 'topo' and unit == 'm':
            unit = u'nm'
            data = data * 1e9
        param.unit = unit
        param.creation_date = dataArray.info['rec_date']
        param.comment = dataArray.info['comment']

        axis1 = AxisParam()
        axis2 = AxisParam()
        axis3 = AxisParam()

        if param.type in ('topo',):
            axis1.length = dataArray.info['scan_pixels'][0]
            axis1.unit = 'nm'
            axis1.increment = dataArray.info['scan_range'][0]/axis1.length*1e9
            axis1.start = 0 # dataArray.info['scan_offset'][0] in meter

            axis2.length = dataArray.info['scan_pixels'][1]
            axis2.unit = 'nm'
            axis2.increment = dataArray.info['scan_range'][1]/axis2.length*1e9
            axis2.start = 0

            param.axis1 = axis1
            param.axis2 = axis2

            print dataArray.info['scan_pixels']
            print dataArray.info['scan_range']

        return Measurement(data, param)

if __name__ == "__main__":
    pass