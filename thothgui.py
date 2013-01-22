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

import platform

from guidata.qt.QtGui import (QMainWindow, QDockWidget, QWidget, QMdiArea,
                              QListWidget, QMessageBox )
from guidata.qt.QtCore import (SIGNAL, SLOT, QObject, Qt, QT_VERSION_STR,
                               PYQT_VERSION_STR, )
from guidata.qthelpers import create_action, add_actions
from guiqwt.config import _

from thoth import Thoth


APP_NAME = _("Thoth")
APP_DESC = _("""Thoth – Scannig Probe Image Analysis Tools""")
VERSION = '0.0.1'
DEMO = False

# Import spyderlib shell, if available
try:
    from spyderlib.widgets.internalshell import InternalShell
    from guidata.qt.QtGui import QFont
    class DockableConsole(InternalShell):
        LOCATION = Qt.BottomDockWidgetArea
        def __init__(self, parent, namespace, message, commands=[]):
            InternalShell.__init__(self, parent=parent, namespace=namespace,
                                   message=message, commands=commands,
                                   multithreaded=False)
            self.setup()

        def setup(self):
            font = QFont("Courier new")
            font.setPointSize(10)
            self.set_font(font)
            self.set_codecompletion_auto(True)
            self.set_calltips(True)
            self.setup_calltips(size=600, font=font)
            self.setup_completion(size=(300, 180), font=font)
except ImportError:
    print "Spyderlib is missing, no console"
    DockableConsole = None


class ThothMainWindow(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.thoth = Thoth()
        self._setup()
        self.file_menu_action = []
        self.console = None

    def closeEvent(self, event):
        if self.console is not None:
            self.console.exit_interpreter()
        event.accept()

    def _setup(self):
        self.setWindowTitle(_('Thoth – Scanning Probe Analysis Tool'))
        self.create_menus()
        self.create_main_frame()
        self.create_status_bar()
        self.create_list_dock()
        self.create_info_dock()
        self.create_internal_console()
        self.connect_thoth()
        self.show()

        if DEMO:
            self.thoth.open('testfiles/flatfile/topo.flat')
            self.thoth.open('testfiles/flatfile/iv.flat')
            self.thoth.open('testfiles/flatfile/grid.flat')

    def connect_thoth(self):
        QObject.connect(self.thoth, SIGNAL("item_registred"),
                        self.thoth.create_window)
        #QObject.connect(self.thoth, SIGNAL("window_registred(QWidget)"), self.mdi_area.addSubWindow)
        QObject.connect(self.thoth, SIGNAL("window_registred"),
                        self.add_sub_window)

    def add_sub_window(self, window):
        self.mdi_area.addSubWindow(window)
        window.show()

    def create_internal_console(self):
        if DockableConsole is None:
            self.console = None
        else:
            import time, scipy.signal as sps, scipy.ndimage as spi
            import sys, os
            import numpy as np
            ns = {'thoth': self.thoth,
                  'np': np, 'sps': sps, 'spi': spi,
                  'os': os, 'sys': sys, 'time': time}
            msg = "Example: thoth.get_items()[0]\n"\
                  "Modules imported at startup: "\
                  "os, sys, os.path as osp, time, "\
                  "numpy as np, scipy.signal as sps, scipy.ndimage as spi"
            self.console = DockableConsole(self, namespace=ns, message=msg)
            console_dock = QDockWidget(_('Console'))
            self.addDockWidget(Qt.BottomDockWidgetArea, console_dock)
            console_dock.setWidget(self.console)
            #self.connect(self.console.interpreter.widget_proxy,
                         #SIGNAL("new_prompt(QString)"),
                         #lambda txt: self.refresh_lists())

    def about(self):
        QMessageBox.about( self, _("About ")+APP_NAME,
              """<b>%s</b> v%s<br>%s<p>%s Pierre Raybaut
              <br>Copyright © François Bianco, University of Geneva
              <br>francois.bianco@unige.ch
              <br>Distributed under the GNU GPL License v.3
              <p>Python %s, Qt %s, PyQt %s %s %s""" % \
              (APP_NAME, VERSION, APP_DESC, _("Developped by"),
               platform.python_version(),
               QT_VERSION_STR, PYQT_VERSION_STR, _("on"), platform.system()) )

    def create_main_frame(self):
        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)

    def create_list_dock(self) :
        list_dock = QDockWidget(_('Files'))
        self.addDockWidget(Qt.RightDockWidgetArea, list_dock)
        self.list_widget = QListWidget()
        #self.connect(list_widget, SIGNAL( "itemSelectionChanged()" ), self.changeSelection)
        list_dock.setWidget(self.list_widget)

    def create_info_dock(self):
        info_dock = QDockWidget(_('Info'))
        self.addDockWidget(Qt.RightDockWidgetArea, info_dock)
        self.info_widget = QListWidget()
        info_dock.setWidget(self.info_widget)

    def create_status_bar(self):
        self.statusBar().showMessage(_("Open a file"))

    def create_menus(self):
        file_menu = self.menuBar().addMenu(_("&File"))

        open_file_action = create_action(self, _("&Open File"),
                                    shortcut="Ctrl+O",
                                    triggered=self.thoth.open,
                                    tip=_("Open a measurement file"))
        #save_plot_action = create_action(self,"&Save all the plots",
            #shortcut="Ctrl+S", triggered=self.save_plots,
            #tip="Save all the plots")
        quit_action = create_action(self, _("&Quit"), triggered=self.close,
            shortcut="Ctrl+Q", tip=_("Close the application"))

        add_actions(file_menu, (open_file_action, None, quit_action))

        help_menu = self.menuBar().addMenu("&Help")
        about_action = create_action(self, _("&About"),
                                     shortcut='F1', triggered=self.about,
                                     tip=_('About Thoth'))

        add_actions(help_menu, (about_action,))


def run():
    from guidata import qapplication
    app = qapplication()
    w = ThothMainWindow()
    app.connect(app, SIGNAL("lastWindowClosed()")
                , app
                , SLOT("quit()")
                )

    app.connect(w, SIGNAL("closed()")
                , app
                , SLOT("quit()")
                )

    w.show()
    app.exec_()

if __name__ == "__main__":
    run()
