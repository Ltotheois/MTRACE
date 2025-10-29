#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Luis Bonah
# Description: Measurement Software for Lab Course

CREDITSSTRING = """Made by Luis Bonah

As this programs GUI is based on PyQt6, which is GNU GPL v3 licensed, this program is also licensed under GNU GPL v3 (See the bottom paragraph).

pandas, matplotlib, scipy and numpy were used for this program, speeding up the development process massively.

Copyright (C) 2020

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

##
## Global Constants and Imports
##
APP_TAG = "MTRACE"

import os
import sys
import re
import time
import pickle
import pyvisa
import serial
import json
import threading
import configparser
import traceback as tb
import numpy as np
import pandas as pd
import webbrowser
import retrophase

from scipy import optimize, special

from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT

QLocale.setDefault(QLocale('en_EN'))
matplotlib.rcParams['axes.formatter.useoffset'] = False

TC_OPTIONS = {0: '10μs', 1: '20μs', 2: '40μs', 3: '80μs', 4: '160μs', 5: '320μs', 6: '640μs', 7: '5ms', 8: '10ms', 9: '20ms', 10: '50ms', 11: '100ms', 12: '200ms', 13: '500ms', 14: '1s', 15: '2s', 16: '5s', 17: '10s', 18: '20s', 19: '50s', 20: '100s', 21: '200s', 22: '500s', 23: '1ks', 24: '2ks', 25: '5ks', 26: '10ks', 27: '20ks', 28: '50ks', 29: '100ks'}
TC_OPTIONS_INV = {value: key for key, value in TC_OPTIONS.items()}
TC_TIMES = {0: 10e-6, 1: 20e-6, 2: 40e-6, 3: 80e-6, 4: 160e-6, 5: 320e-6, 6: 640e-6, 7: 5e-3, 8: 10e-3, 9: 20e-3, 10: 50e-3, 11: 100e-3, 12: 200e-3, 13: 500e-3, 14: 1, 15: 2, 16: 5, 17: 10, 18: 20, 19: 50, 20: 100, 21: 200, 22: 500, 23: 1e3, 24: 2e3, 25: 5e3, 26: 10e3, 27: 20e3, 28: 50e3, 29: 100e3}

SEN_OPTIONS = {
	1: '2nV', 2: '5nV', 3: '10nV', 4: '20nV', 5: '50nV', 6: '100nV', 7: '200nV', 8: '500nv',
	9: '1µV', 10: '2µV', 11: '5µV', 12: '10µV', 13: '20µV', 14: '50µV', 15: '100µV', 16: '200µV', 17: '500µV', 
	18: '1mV', 19: '2mV', 20: '5mV', 21: '10mV', 22: '20mV', 23: '50mV', 24: '100mV', 25: '200mV', 26: '500mV', 27: '1V'
}
SEN_OPTIONS_INV = {value: key for key, value in SEN_OPTIONS.items()}

PRESSURE_GAUGE_KWARGS = {'baudrate': 9600, 'bytesize': 8, 'stopbits': 1, 'parity': serial.PARITY_NONE, 'timeout': 0.5}

##
## Global Decorators
##
class AtomicCounter():
	def __init__(self, initial=0, callback=None):
		self.value = initial
		self._lock = threading.Lock()
		
		if callback is not None:
			self.callback = callback

	def increase(self, num=1):
		with self._lock:
			self.value += num
			return self.value

	def decrease(self, num=1):
		return(self.increase(-num))

	def get_value(self):
		with self._lock:
			return(self.value)

	def __enter__(self, *args):
		self.increase()

	def __exit__(self, type, value, traceback):
		if self.decrease() == 0:
			self.callback()

	def callback(self):
		pass

class QThread(QThread):
	threads = set()

	@classmethod
	def threaded_d(cls, func):
		def wrapper(*args, **kwargs):
			thread = cls(func, *args, **kwargs)
			thread.start()
			return(thread)
		return(wrapper)

	def __init__(self, function, *args, **kwargs):
		super().__init__()
		self.function, self.args, self.kwargs = function, args, kwargs
		self.threads.add(self)
		self.finished.connect(lambda x=self: self.threads.remove(x))

	def run(self):
		self.function(*self.args, **self.kwargs)

class MeasurementSoftware(QApplication):
	configsignal = pyqtSignal(tuple)
	
	def __init__(self, *args, **kwargs):
		sys.excepthook = except_hook
		threading.excepthook = lambda args: except_hook(*args[:3])
		
		super().__init__(sys.argv, *args, **kwargs)
		self.setStyle("Fusion")

		Geometry().load()

		global config
		config = Config(self.configsignal)
		config.load()
		messages = config.messages
		
		self.initialize_matplotlib_settings()

		with config:
			global mainwindow
			mainwindow = MainWindow(self)
			mainwindow.create_gui_components()
			mainwindow.show()
			
			if messages:
				notify_warning.emit('\n'.join(messages))

		sys.exit(self.exec())

	def initialize_matplotlib_settings(self):
		self.styleHints().colorSchemeChanged.connect(self.update_matplotlib_theme)
		self.update_matplotlib_theme()
		
		mpl_style_filename = llwpfile('.mplstyle')
		if os.path.exists(mpl_style_filename):
			matplotlib.style.use(mpl_style_filename)

	def update_matplotlib_theme(self):
		matplotlib.style.use('dark_background' if is_dark_theme() else 'default')
		matplotlib.rcParams['figure.facecolor'] = '#00000000'
		matplotlib.rcParams['axes.facecolor'] = '#00000000'

class MainWindow(QMainWindow):
	notify_info = pyqtSignal(str)
	notify_warning = pyqtSignal(str)
	notify_error = pyqtSignal(str)

	status_counter = AtomicCounter()
	working = pyqtSignal()
	waiting = pyqtSignal()


	def __init__(self, app, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.setWindowTitle(APP_TAG)
		self.setAcceptDrops(True)
		Geometry.load_widget_geometry(self)
		
		try:
			# Set logo as icon
			possible_folders = [os.path.dirname(os.path.realpath(__file__)), os.getcwd()]
			for folder in possible_folders:
				iconpath = os.path.join(folder, "MINI_TRACE.svg")
				if os.path.isfile(iconpath):
					icon = QIcon(iconpath)
					break
			
			# Make it appear separate from python scripts in taskbar
			app.setWindowIcon(QIcon(iconpath))
			import ctypes
			ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_TAG)
		except Exception as E:
			pass
		
	def create_gui_components(self):
		global notify_info, notify_warning, notify_error
		notify_info = self.notify_info
		notify_warning = self.notify_warning
		notify_error = self.notify_error

		self.statusbar = StatusBar()
		self.setStatusBar(self.statusbar)

		ConfigWindow()
		LogWindow()
		PressureWindow()
		
		self.mainwidget = MainWidget()
		self.setCentralWidget(self.mainwidget)
		self.menu = Menu(self)

	def togglefullscreen(self):
		if self.isFullScreen():
			self.showNormal()
		else:
			self.showFullScreen()

	def moveEvent(self, *args, **kwargs):
		Geometry.save_widget_geometry(self)
		return super().moveEvent(*args, **kwargs)

	def resizeEvent(self, *args, **kwargs):
		Geometry.save_widget_geometry(self)
		return super().resizeEvent(*args, **kwargs)

	def closeEvent(self, *args, **kwargs):
		Geometry.set("__dockstate__", self.saveState())
		Geometry.save()
		config.save()


class MainWidget(QGroupBox):
	drawplot = pyqtSignal()
	newdata_available = pyqtSignal()
	measurement_finished = pyqtSignal()
	draw_counter = AtomicCounter()

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		self.freqs = np.zeros((0,))
		self.xs = np.zeros((0,))
		self.ys = np.zeros((0,))
		self.pressure_before = None
		self.pressure_after = None
		self.values = None
		self.nextfrequency_counter = AtomicCounter()
		self.state = 'waiting'
		
		self.fit_vline = None
		self.fit_curve = None

		self.lockin = None
		self.synthesizer = None

		layout = QVBoxLayout()
		self.setLayout(layout)
		
		self.fig = matplotlib.figure.Figure(dpi=config["plot_dpi"])
		self.ax = self.fig.subplots()
		self.exp_coll = matplotlib.collections.LineCollection(np.zeros(shape=(0,2,2)), colors=config["color_exp"], capstyle='round')
		self.ax.add_collection(self.exp_coll)

		self.plotcanvas = FigureCanvas(self.fig)
		layout.addWidget(self.plotcanvas, 6)
		
		self.mpltoolbar = NavigationToolbar2QT(self.plotcanvas, self)
		self.mpltoolbar.setVisible(config["flag_matplotlibtoolbar"])
		config.register('flag_matplotlibtoolbar', lambda: self.mpltoolbar.setVisible(config['flag_matplotlibtoolbar']))
		layout.addWidget(self.mpltoolbar)

		tmp_layout = QGridLayout()
		
		i_row = 0
		tmp_layout.addWidget(QQ(QLabel, text='Frequency Center [MHz]: '), i_row, 0)
		tmp_layout.addWidget(QQ(QDoubleSpinBox, 'measurement_center', range=(0, np.inf)), i_row, 1)

		i_row += 1
		tmp_layout.addWidget(QQ(QLabel, text='Frequency Span [MHz]: '), i_row, 0)
		tmp_layout.addWidget(QQ(QDoubleSpinBox, 'measurement_span', range=(0, np.inf)), i_row, 1)

		i_row += 1
		tmp_layout.addWidget(QQ(QLabel, text='Frequency Stepsize [kHz]: '), i_row, 0)
		tmp_layout.addWidget(QQ(QDoubleSpinBox, 'measurement_stepsize', range=(0, np.inf)), i_row, 1)

		i_row += 1
		tmp_layout.addWidget(QQ(QLabel, text='Repetitions: '), i_row, 0)
		tmp_layout.addWidget(QQ(QSpinBox, 'measurement_repetitions', range=(0, None)), i_row, 1)

		i_row += 1
		tmp_layout.addWidget(QQ(QLabel, text='Notes: '), i_row, 0)
		tmp_layout.addWidget(QQ(QLineEdit, 'measurement_notes'), i_row, 1)

		i_row = 0
		tmp = ('AM', '1f-FM', '2f-FM')
		tmp_layout.addWidget(QQ(QLabel, text='FM/AM Mode: '), i_row, 3)
		tmp_layout.addWidget(QQ(QComboBox, 'measurement_modulationtype', options=tmp), i_row, 4)

		i_row += 1
		tmp_layout.addWidget(QQ(QLabel, text='FM/AM Frequency [kHz]: '), i_row, 3)
		tmp_layout.addWidget(QQ(QDoubleSpinBox, 'measurement_modulationfrequency', range=(0, np.inf)), i_row, 4)

		i_row += 1
		tmp_layout.addWidget(QQ(QLabel, text='FM Deviation [kHz]: '), i_row, 3)
		tmp_layout.addWidget(QQ(QDoubleSpinBox, 'measurement_modulationdeviation', range=(0, np.inf)), i_row, 4)

		i_row += 1
		tmp_layout.addWidget(QQ(QLabel, text='Integration Time: '), i_row, 3)
		tmp_layout.addWidget(QQ(QComboBox, 'measurement_timeconstant', options=TC_OPTIONS.values()), i_row, 4)


		i_row += 1
		tmp_layout.addWidget(QQ(QLabel, text='Lock-In Sensitivity: '), i_row, 3)
		tmp_layout.addWidget(QQ(QComboBox, 'measurement_lockinsensitivity', options=SEN_OPTIONS.values()), i_row, 4)

		i_row = 0
		self.start_button = QQ(QPushButton, text='Start', change=self.start_measurement)
		self.abort_button = QQ(QPushButton, text='Abort', change=self.abort_measurement, hidden=True)
		tmp_layout.addWidget(self.start_button, i_row, 6)
		tmp_layout.addWidget(self.abort_button, i_row, 6)

		i_row += 1
		self.pause_button = QQ(QPushButton, text='Pause', change=self.pause_measurement)
		self.continue_button = QQ(QPushButton, text='Continue', change=self.continue_measurement, hidden=True)
		tmp_layout.addWidget(self.pause_button, i_row, 6)
		tmp_layout.addWidget(self.continue_button, i_row, 6)

		i_row += 1
		self.save_button = QQ(QPushButton, text='Save', change=lambda x: self.save_measurement())
		tmp_layout.addWidget(self.save_button, i_row, 6)

		tmp_layout.setColumnMinimumWidth(1, 150)
		tmp_layout.setColumnMinimumWidth(2, 80)
		tmp_layout.setColumnMinimumWidth(5, 80)
		tmp_layout.setColumnMinimumWidth(4, 150)
		tmp_layout.setColumnStretch(7, 1)
		layout.addLayout(tmp_layout)
		layout.addStretch()
		
		self.newdata_available.connect(self.update_plot)
		self.measurement_finished.connect(self.after_measurement)
		self.drawplot.connect(self.draw_canvas)
		self.plotcanvas.draw()
		
		self.listener_onhover = self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
		self.span_selector = matplotlib.widgets.SpanSelector(self.ax, self.on_range, 'horizontal', useblit=True)

	def update_plot(self):
		tmp_xs = self.freqs
		
		component = config['plot_ycomponent']
		if component == 'X':
			tmp_ys = self.xs
		elif component == 'Y':
			tmp_ys = self.ys
		else:
			tmp_ys = np.sqrt(np.power(self.xs, 2), np.power(self.ys, 2))

		mask = (~np.isnan(tmp_ys))
		tmp_xs, tmp_ys = tmp_xs[mask], tmp_ys[mask]
		
		if not len(tmp_xs):
			segs = np.array([])
			yrange = (0, 1)
		else:
			segs = np.array(((tmp_xs[:-1], tmp_xs[1:]), (tmp_ys[:-1], tmp_ys[1:]))).T
			yrange = np.nanmin(tmp_ys), np.nanmax(tmp_ys)
			
			margin = config['plot_ymargin']
			yrange = (yrange[0]-margin*(yrange[1]-yrange[0]), yrange[1]+margin*(yrange[1]-yrange[0]))
			if yrange[0] == yrange[1]:
				yrange = (yrange[0]-1, yrange[0]+1)
			
		color = config['color_exp']
		self.exp_coll.set(segments=segs, color=color)
		self.ax.set_ylim(yrange)
		self.drawplot.emit()

	def start_measurement(self):
		self.state = 'running'
		self.start_button.setHidden(True)
		self.abort_button.setHidden(False)
		self.save_button.setDisabled(True)

		values = config.copy()
		self.run_measurement(values)

	def abort_measurement(self):
		self.state = 'aborting'

	def pause_measurement(self):
		if self.state != 'running':
			return
		self.state = 'paused'
		self.continue_button.setHidden(False)
		self.pause_button.setHidden(True)

	def continue_measurement(self):
		self.state = 'running'
		self.continue_button.setHidden(True)
		self.pause_button.setHidden(False)

	
	def after_measurement(self):
		self.state = 'waiting'
		self.start_button.setHidden(False)
		self.abort_button.setHidden(True)
		self.continue_button.setHidden(True)
		self.pause_button.setHidden(False)
		self.save_button.setDisabled(False)

		# Save file into MTRACE folder
		homefolder = llwpfile()
		directory = os.path.join(homefolder, "data", time.strftime("%Y%m%d", time.localtime()))
		if not os.path.exists(directory):
			os.makedirs(directory, exist_ok=True)
		fname = os.path.join(directory, time.strftime("%H-%M-%S", time.localtime()))
		self.save_measurement(fname)

	@QThread.threaded_d
	def run_measurement(self, values):
		try:
			self.values = values
			self.pressure_after = None
			self.pressure_before = None

			# Connect devices
			rm = pyvisa.ResourceManager()
			self.synthesizer = synthesizer = rm.open_resource(values['address_synthesizer'])
			self.lockin = lockin = rm.open_resource(values['address_lockin'])
			freq_mult = values['measurement_frequencymultiplication']
			
			# Calculate frequency positions
			center, span = values['measurement_center'], values['measurement_span']
			stepsize = values['measurement_stepsize'] / 1000
			self.ax.set_xlim(center - span/freq_mult, center + span/freq_mult)
			self.remove_fitline()
			self.drawplot.emit()
			N = span/freq_mult // stepsize + 1
			self.freqs = np.arange(-N, N+1) * stepsize + center

			# Repetitions
			if values['measurement_repetitions'] > 1:
				tmp = []
				for i in range(values['measurement_repetitions']):
					if i % 2 == 0:
						tmp.append(self.freqs)
					else:
						tmp.append(self.freqs[::-1])
				self.freqs = np.concatenate(tmp)
			
			# Divide by two to take into account the doubler
			fm_deviation = values['measurement_modulationdeviation'] / freq_mult

			# Determin OSC Amplitude
			if values['measurement_modulationtype'] in ('1f-FM', '2f-FM'):
				fm_factors = {7: 10000, 6: 3000, 5: 1000, 4: 300, 3: 100, 2: 30}
				tmp = {key: value for key, value in fm_factors.items() if value >= fm_deviation}
				fm_factor_key = min(tmp, key=tmp.get)
				fm_factor = fm_factors[fm_factor_key]		
				osc_amp = fm_deviation / fm_factor
			else:
				osc_amp = 0.5

			# Delay Time
			tc_key = TC_OPTIONS_INV[values['measurement_timeconstant']]
			delay_time = TC_TIMES[tc_key] * 2 + values['measurement_additionaldelaytime']

			# Set up Lock-In Amplifier
			startvalues = {
				"IE": 		0,
				"REFN": 	2 if values['measurement_modulationtype'] == '2f-FM' else 1,
				"VMODE": 	1,
				"SLOPE": 	0,
				"TC": 		tc_key,
				"OF.":		f'{values["measurement_modulationfrequency"]*1e3:.4f}',
				"OA.":		f'{osc_amp / np.sqrt(2):.4f}',
				"SEN": 		SEN_OPTIONS_INV[values['measurement_lockinsensitivity']],
				"ACGAIN": 	values['measurement_acgain'],
			}

			if not values['measurement_skipinitialization']:
				for key, value in startvalues.items():
					lockin.write(f"{key} {value}")
				lockin.query("*OPC?")

				# Set up Synthesizer
				synthesizer.write('R0') # RF off
				synthesizer.write('A0') # AM off
				synthesizer.write('D0') # FM off
				synthesizer.write('P0') # PM off
				synthesizer.write('W0') # Sweep off

				# Set this up conditionally, depending on the AM/FM mode
				if values['measurement_modulationtype'] in ('1f-FM', '2f-FM'):
					synthesizer.write(f'D{fm_factor_key:1.0f}') # 0.03MHz FM
				else:
					synthesizer.write('A2') # 30% AM

			freq = self.freqs[0]
			synthesizer.write(f'FR{freq/freq_mult*1E6}HZ')
			synthesizer.write('RA13DB') # Set to full power
			synthesizer.write('R1') # RF on

			time.sleep(0.5)
			self.pressure_before = measure_pressure(values['address_pressuregauge'], values['measurement_skippressure'])

			self.xs = np.full_like(self.freqs, np.nan)
			self.ys = self.xs.copy()
			for i, freq in enumerate(self.freqs):
				while self.state != 'running':
					if self.state == 'aborting':
						return
					
					tmp = self.nextfrequency_counter.get_value()
					if tmp > 0:
						self.nextfrequency_counter.decrease(tmp)
						break
					
					time.sleep(0.1)
				
				synthesizer.write(f'FR{freq/freq_mult*1E6}HZ')
			
				counterstart = time.perf_counter()
				# Sleep to not hog the CPU
				time.sleep(delay_time)
				while time.perf_counter() - counterstart < delay_time:
					continue

				tmp = lockin.query("XY.?")
				x, y = [float(x.split("\x00")[0]) for x in tmp.split(",")]
				self.xs[i], self.ys[i] = x*1e6, y*1e6
				self.newdata_available.emit()

			freq = center
			synthesizer.write(f'FR{freq/freq_mult*1E6}HZ') # Set frequency to center frequency
			synthesizer.write('R0') # RF off
			
			if values['flag_autophase']:
				fs, xs, ys = self.freqs, self.xs, self.ys
				rabss = np.sqrt(xs**2 + ys**2)
				std = np.std(rabss)
				xmax = np.max(rabss)
			
				if std / xmax > 0.1:
					notify_warning.emit('Found no strong signal in the measurement, therefore the phase was not automatically adjusted.')
				
				else:
					best_phase = retrophase.autophase(fs, xs, ys)
					new_xs, new_ys = retrophase.change_phase(fs, xs, ys, best_phase)
					
					if best_phase < 0:
						best_phase += 360
					notify_info.emit(f'Optimized the phase by a shift of {best_phase:-6.2f}°.')
					
					self.xs, self.ys = new_xs, new_ys
					self.newdata_available.emit()

			self.pressure_after = measure_pressure(values['address_pressuregauge'], values['measurement_skippressure'])

		except Exception as E:
			raise E
		finally:
			self.measurement_finished.emit()
			
			if self.lockin:
				self.lockin.close()
				self.lockin = None
			
			if self.synthesizer:
				self.synthesizer.close()
				self.synthesizer = None

	def save_measurement(self, fname=None):
		if fname is None:
			fname = QFileDialog.getSaveFileName(None, 'Choose file to save measurement to',"","CSV Files (*.csv);;All Files (*)")[0]
			if not fname or self.values is None:
				return
		
		fs, xs, ys = self.freqs.copy(), self.xs.copy(), self.ys.copy()
		df = pd.DataFrame({'Frequency': fs, 'X': xs, 'Y': ys})
		data = df.groupby('Frequency').mean().reset_index().values

		header = '\n'.join([
			f'FM/AM Mode: {self.values["measurement_modulationtype"]}',
			f'FM/AM Frequency: {self.values["measurement_modulationfrequency"]}',
			f'FM/AM Deviation: {self.values["measurement_modulationdeviation"]}',
			f'Timeconstant: {self.values["measurement_timeconstant"]}',
			f'Repetitions: {self.values["measurement_repetitions"]}',
			f'Lock-In Sensitiviy: {self.values["measurement_lockinsensitivity"]}',
			f'Pressure before: {self.pressure_before}',
			f'Pressure after: {self.pressure_after}',
			f'Notes: {self.values["measurement_notes"]}',
		])
		np.savetxt(fname, data, delimiter="\t", header=header)

	def autophase_lockin(self):
		if self.lockin:
			self.lockin.write('AQN')
			self.lockin.query('*OPC?')
		else:
			notify_warning.emit('No lockin available, cannot optimize the phase.')

	def draw_canvas(self):
		self.plotcanvas.draw_idle()

	def on_hover(self, event):
		x = event.xdata
		y = event.ydata

		if not all([x, y, event.inaxes]):
			text_cursor = ""
		else:
			text_cursor = f"  ({x=:{config['flag_xformatfloat']}}, {y=:{config['flag_xformatfloat']}})  "
		mainwindow.statusbar.position_label.setText(text_cursor)

	def remove_fitline(self):
		if self.fit_vline is not None:
			self.fit_vline.remove()
			self.fit_vline = None
		if self.fit_curve is not None:
			self.fit_curve.remove()
			self.fit_curve = None
	
	def on_range(self, xmin, xmax):
		if xmax == xmin:
			return
			
		self.remove_fitline()
		tmp_xs = self.freqs
		
		component = config['plot_ycomponent']
		if component == 'X':
			tmp_ys = self.xs
		elif component == 'Y':
			tmp_ys = self.ys
		else:
			tmp_ys = np.sqrt(np.power(self.xs, 2), np.power(self.ys, 2))

		mask = (tmp_xs > xmin) & (tmp_xs < xmax) & (~np.isnan(tmp_ys))
		tmp_xs, tmp_ys = tmp_xs[mask], tmp_ys[mask]

		fitmethod = config['fit_fitmethod']
		peakdirection = config['fit_peakdirection']

		if (len(tmp_xs) == 0) or ((len(tmp_xs) < 2) and fitmethod != 'Pgopher'):
			notify_error.emit('The data could not be fit as there were too few points selected.')
			raise GUIAbortedError('The data could not be fit as there were too few points selected.')

		fit_xs = np.linspace(xmin, xmax, config['fit_xpoints'])

		try:
			fit_function = get_fitfunction(fitmethod, config['fit_offset'])
			xmiddle, xuncert, fit_xs, fit_ys = fit_function(tmp_xs, tmp_ys, peakdirection, fit_xs)
		except Exception as E:
			notify_error.emit(f"The fitting failed with the following error message : {str(E)}")
			raise
		
		if config['fit_copytoclipboard']:
			QApplication.clipboard().setText(str(xmiddle))

		# Highlight fit in plot
		self.fit_curve = self.ax.plot(fit_xs, fit_ys, color=config["color_fit"], alpha=0.7, linewidth=1)[0]
		self.fit_vline = self.ax.axvline(x=xmiddle, color=config["color_fit"], ls="--", alpha=1, linewidth=1)

		self.drawplot.emit()

def llwpfile(extension=""):
	home = os.path.expanduser("~")
	llwpfolder = os.path.join(home, f".{APP_TAG.lower()}")
	
	if not os.path.isdir(llwpfolder):
		os.mkdir(llwpfolder)

	return(os.path.join(llwpfolder, extension))

def is_dark_theme():
	return(QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark)

# Fitting
def Gaussian(derivative, x, x0, amp, fwhm):
	sigma = fwhm/(2*np.sqrt(2*np.log(2)))
	if sigma == 0:
		return [0 if i!=x0 else np.inf for i in x]
	
	if derivative == 0:
		ys = amp * np.exp(-(x-x0)**2/(2*sigma**2))
	elif derivative == 1:
		ys = -amp / (sigma * np.exp(-0.5)) * (x-x0) * np.exp(-(x-x0)**2/(2*sigma**2))
	elif derivative == 2:
		ys = amp * (1 - ((x-x0)/sigma)**2) * np.exp(-(x-x0)**2/(2*sigma**2))
	else:
		raise NotImplementedError('Only the zeroth, first, and second derivatives of a Gaussian are implemented.')
	return(ys)

def Lorentzian(derivative, x, x0, amp, fwhm):
	gamma = fwhm/2
	if gamma == 0:
		return [0 if i!=x0 else np.inf for i in x]
	
	if derivative == 0:
		ys = gamma**2 /((gamma**2 + (x-x0)**2))
	elif derivative == 1:
		ys = (-amp*gamma**3 * 16/9 * np.sqrt(3)) * (x-x0)/((x-x0)**2 + gamma**2)**2
	elif derivative == 2:
		ys = (amp * gamma**4) * (gamma**2 - 3 * (x-x0)**2) / ((x-x0)**2 + gamma**2)**3
	else:
		raise NotImplementedError('Only the zeroth, first, and second derivatives of a Gaussian are implemented.')
	return(ys)

def Voigt(derivative, x, x0, amp, fwhm_gauss, fwhm_lorentz):
	sigma = fwhm_gauss/(2*np.sqrt(2*np.log(2)))
	gamma = fwhm_lorentz/2
	if gamma == sigma == 0:
		return [0 if i!=x0 else np.inf for i in x]
	
	z = (x - x0 + 1j * gamma) / (sigma * np.sqrt(2))
	wz = special.wofz(z)
	w0 = special.wofz((1j * gamma) / (sigma * np.sqrt(2)))
	
	if derivative == 0:
		tmp = lambda x, x0, wz, sigma, gamma: np.real(wz)/(sigma * np.sqrt(2*np.pi))            
	elif derivative == 1:
		tmp = lambda x, x0, wz, sigma, gamma: 1/(sigma**3 * np.sqrt(2*np.pi)) * (gamma * np.imag(wz) - (x-x0) * np.real(wz))
	elif derivative == 2:
		tmp = lambda x, x0, wz, sigma, gamma: 1/(sigma**5 * np.sqrt(2*np.pi)) * (gamma * (2*(x-x0) * np.imag(wz) - sigma * np.sqrt(2/np.pi)) + (gamma**2 + sigma**2 - (x-x0)**2) * np.real(wz))
	else:
		raise NotImplementedError('Only the zeroth, first, and second derivatives of a Gaussian are implemented.')
	
	ys = tmp(x, x0, wz, sigma, gamma)
	ymax = tmp(0, 0, w0, sigma, gamma)
	ys *= amp / ymax

	return(ys)

def lineshape(shape, derivative, *args):
	if shape == 'Gauss':
		lineshape_function = Gaussian
	elif shape == 'Lorentz':
		lineshape_function = Lorentzian
	elif shape == 'Voigt':
		lineshape_function = Voigt
	else:
		raise NotImplementedError(f'The lineshape {shape} is not implemented.')
	
	ys = lineshape_function(derivative, *args)
	return(ys)

def fit_pgopher(xs, ys, peakdirection, fit_xs):
	ymin, ymax = np.min(ys), np.max(ys)

	if peakdirection < 0:
		cutoff = ymin + (ymax-ymin)/2
		mask = (ys <= cutoff)
	else:
		cutoff = ymax - (ymax-ymin)/2
		mask = (ys >= cutoff)

	fit_xs = xs[mask]
	fit_ys = ys[mask] - ymin

	xmiddle = np.sum(fit_xs*fit_ys)/np.sum(fit_ys)
	xuncert = 0

	return(xmiddle, xuncert, fit_xs, fit_ys + ymin)

def fit_polynom(xs, ys, peakdirection, fit_xs, rank):
	try:
		popt = np.polyfit(xs, ys, rank)
	except Exception as E:
		popt = np.polyfit(xs, ys, rank)
	polynom = np.poly1d(popt)
	fit_ys = polynom(fit_xs)

	if peakdirection < 0:
		xmiddle = fit_xs[np.argmin(fit_ys)]
	else:
		xmiddle = fit_xs[np.argmax(fit_ys)]

	xuncert = 0
	return(xmiddle, xuncert, fit_xs, fit_ys)

def fit_polynom_multirank(xs, ys, peakdirection, fit_xs, maxrank):
	best_rms = np.inf
	best_rank = 0

	maxrank = min(len(xs), maxrank)
	for rank in range(maxrank):
		try:
			try:
				popt = np.polyfit(xs, ys, rank)
			except Exception as E:
				popt = np.polyfit(xs, ys, rank)
			polynom = np.poly1d(popt)
			fit_ys = polynom(exp_xs)

			rms = np.mean((fit_ys - ys)**2)
			if rms < best_rms:
				best_rms = rms
				best_rank = rank
		except Exception as E:
			continue

	popt = np.polyfit(xs, ys, best_rank)
	polynom = np.poly1d(popt)
	fit_ys = polynom(fit_xs)

	if peakdirection < 0:
		xmiddle = fit_xs[np.argmin(fit_ys)]
	else:
		xmiddle = fit_xs[np.argmax(fit_ys)]

	xuncert = 0
	return(xmiddle, xuncert, fit_xs, fit_ys)

def fit_lineshape(xs, ys, peakdirection, fit_xs, profilename, derivative, offset, **kwargs):
	xmin, xmax = xs.min(), xs.max()
	x0 = (xmin + xmax) / 2
	
	xs_weight_factor = kwargs.get('xs_weight_factor', 4)
	if xs_weight_factor:
		ys_weighted = ys * np.exp(- np.abs(np.abs(xs - x0) / (xmax - xmin) ) * xs_weight_factor)
		x0 = xs[np.argmax(ys_weighted)] if peakdirection >= 0 else xs[np.argmin(ys_weighted)]
	
	ymin, ymax, ymean, yptp = ys.min(), ys.max(), ys.mean(), np.ptp(ys)
	y0 = 0
	
	w0 = kwargs.get('w0', (xmax - xmin) / 10 )
	wmin = kwargs.get('wmin', 0)
	wmax = kwargs.get('wmax', (xmax - xmin) )
	
	amp_min, amp_max = -3*yptp, 3*yptp
	if peakdirection < 0:
		amp_max = 0
		y0 = -yptp
	if peakdirection > 0:
		amp_min = 0
		y0 = yptp

	p0 = [x0, y0, w0] if profilename != 'Voigt' else [x0, y0, w0, w0]
	bounds = [[xmin, amp_min, wmin], [xmax, amp_max, wmax]] if profilename != 'Voigt' else [[xmin, amp_min, wmin, wmin], [xmax, amp_max, wmax, wmax]]
	function = lambda *x: lineshape(profilename, derivative, *x)

	if offset:
		function = lambda *x: lineshape(profilename, derivative, *x[:-1]) + x[-1]
		p0.append(ymean)
		bounds[0].append(ymin)
		bounds[1].append(ymax)

	try:
		popt, pcov = optimize.curve_fit(function, xs, ys, p0=p0, bounds=bounds)
	except Exception as E:
		popt, pcov = optimize.curve_fit(function, xs, ys, p0=p0, bounds=bounds)
	perr = np.sqrt(np.diag(pcov))
	fit_ys = function(fit_xs, *popt)

	xmiddle = popt[0]
	xuncert = perr[0]

	filename = llwpfile(".fit")
	datetime = time.strftime("%d.%m.%Y %H:%M:%S", time.localtime())
	
	pressure_before = mainwindow.mainwidget.pressure_before
	pressure_after = mainwindow.mainwidget.pressure_after

	notes = config['measurement_notes']
	spacer = '#' * 46
	header = f'Fit from {datetime}\nProfile: {profilename} {derivative}-Derivative\nPressure before: {pressure_before}\nPressure after: {pressure_after}\nNotes: {notes}\n\n'
	labels = ('Center', 'Amplitude', 'FWHM') if profilename != 'Voigt' else ('Center', 'Amplitude', 'FWHM Gauss', 'FWHM Lorentz')
	
	message = f'{spacer}\n{header}'
	for value, uncert, label in zip(popt, perr, labels):
		tmp_label = label + ':'
		message += f'{tmp_label:13} {value:14.6f} ± {uncert:14.6f}\n'
	message += f'{spacer}\n\n\n'

	with open(filename, "a+") as file:
		file.write(message)
	
	return(xmiddle, xuncert, fit_xs, fit_ys)

def get_fitfunction(fitmethod, offset=False, **kwargs):
	fit_function = {
		'Pgopher': fit_pgopher,
		'Polynom': lambda *args: fit_polynom(*args, config['fit_polynomrank']),
		'MultiPolynom': lambda *args: fit_polynom_multirank(*args, config['fit_polynommaxrank']),
	}.get(fitmethod)
	
	if not fit_function:
		profilename, derivative = {
			'Gauss':					('Gauss', 0),
			'Lorentz':					('Lorentz', 0),
			'Voigt':					('Voigt', 0),
			'Gauss 1st Derivative':		('Gauss', 1),
			'Lorentz 1st Derivative':	('Lorentz', 1),
			'Voigt 1st Derivative':		('Voigt', 1),
			'Gauss 2nd Derivative':		('Gauss', 2),
			'Lorentz 2nd Derivative':	('Lorentz', 2),
			'Voigt 2nd Derivative':		('Voigt', 2),
		}[fitmethod]
		fit_function = lambda *args, kwargs=kwargs: fit_lineshape(*args, profilename, derivative, offset, **kwargs)
	return(fit_function)



class Menu():
	def __init__(self, parent, *args, **kwargs):
		mb = parent.menuBar()
		
		# Create top level menus
		top_menu_labels = ("View", "Actions", "Fit")
		self.top_menus = {}
		for label in top_menu_labels:
			menu = mb.addMenu(f'{label}')
			self.top_menus[label] = menu


		toggleaction_config = ConfigWindow.instance.toggleViewAction()
		toggleaction_config.setShortcut('Shift+0')

		toggleaction_log = LogWindow.instance.toggleViewAction()
		toggleaction_log.setShortcut('Shift+1')

		toggleaction_pressure = PressureWindow.instance.toggleViewAction()
		toggleaction_pressure.setShortcut('Shift+2')

		plot_component_menu = QMenu('Plot y-data', parent=parent)
		current_component = config['plot_ycomponent']
		self.component_actions = {}
		for component in ('X', 'Y', 'R'):
			is_checked = (component == current_component)
			callback = lambda _, component=component: self.set_component(component)
			self.component_actions[component] = QQ(QAction, parent=parent, text=f"{component}", change=callback, checkable=True, value=is_checked)
			plot_component_menu.addAction(self.component_actions[component])
		config.register('plot_ycomponent', self.on_component_changed)

		fitfunction_menu = QMenu("Choose Fit Function", parent=parent)
		self.fitfunction_actions = {}

		current_method = config['fit_fitmethod']
		for method in ('Pgopher', 'Polynom', 'MultiPolynom', 'Gauss', 'Lorentz',
				'Voigt', 'Gauss 1st Derivative', 'Lorentz 1st Derivative', 'Voigt 1st Derivative',
				'Gauss 2nd Derivative', 'Lorentz 2nd Derivative', 'Voigt 2nd Derivative', ):
			is_checked = (method == current_method)
			callback = lambda _, method=method: self.set_fitmethod_gui(method)
			self.fitfunction_actions[method] = QQ(QAction, parent=parent, text=f"{method}", change=callback, checkable=True, value=is_checked)
			fitfunction_menu.addAction(self.fitfunction_actions[method])
		config.register('fit_fitmethod', self.on_fitfunction_changed)

		actions = {
			'Actions': (
				QQ(QAction, parent=parent, text="Save current values as default", tooltip="Save current configuration as default", change=lambda _: config.save()),
				QQ(QAction, parent=parent, text="Open MTRACE folder", change=lambda x: webbrowser.open(f'file:///{llwpfile()}'), tooltip="Open the folder containing the config, ...", ),
				None,
				QQ(QAction, parent=parent, text='Next Frequency', change=lambda x: mainwindow.mainwidget.nextfrequency_counter.increase(), shortcut='Ctrl+N'),
				None,
				QQ(QAction, parent=parent, text='Autophase', change=lambda x: mainwindow.mainwidget.autophase_lockin()),
				None,
				QQ(QAction, parent=parent, text='Create testdata', change=lambda x: self.show_test_data()),
			),
			'View': (
				toggleaction_config,
				toggleaction_log,
				toggleaction_pressure,
				QQ(QAction, 'flag_matplotlibtoolbar', checkable=True, text='MPL Toolbar'),
				None,
				plot_component_menu,
				None,
			),
			'Fit': (
				fitfunction_menu,
				None,
				QQ(QAction, parent=parent, text="Change Fit Color", tooltip="Change the color of the fitfunction", change=lambda _: self.change_fitcolor()),
				None,
				QQ(QAction, parent=parent, text="Show Fit Results", tooltip="Show the path to the file holding the results from fitting the lineshapes", change=lambda _: self.show_fit_file()),
				
			),
		}

		for label, menu in self.top_menus.items():
			for widget in actions.get(label, []):
				if widget is None:
					menu.addSeparator()
				elif isinstance(widget, QAction):
					menu.addAction(widget)
				else:
					menu.addMenu(widget)
	
	def set_component(self, component):
		if component == config['plot_ycomponent']:
			self.on_component_changed()
		else:
			config['plot_ycomponent'] = component
	
	def on_component_changed(self):
		value = config["plot_ycomponent"]
		for component, action in self.component_actions.items():
			action.setChecked(component == value)
		
		mainwindow.mainwidget.newdata_available.emit()
	
	def set_fitmethod_gui(self, method):
		if method == config['fit_fitmethod']:
			self.on_fitfunction_changed()
		else:
			config['fit_fitmethod'] = method

	def on_fitfunction_changed(self):
		value = config["fit_fitmethod"]
		for fitfunction, action in self.fitfunction_actions.items():
			action.setChecked(fitfunction == value)
		notify_info.emit(f"Fitting with method '{value}'")

	def change_fitcolor(self):
		color = QColorDialog.getColor(initial=QColor(Color.rgbt_to_trgb(config['color_fit'])), options=QColorDialog.ColorDialogOption.ShowAlphaChannel)
		color = Color(Color.trgb_to_rgbt(color.name(QColor.NameFormat.HexArgb)))
		config['color_fit'] = color

	def show_test_data(self):
		xrange = 26990, 27010
		xs = np.linspace(*xrange, 10000)
		ys = np.zeros_like(xs)
		
		ys += Voigt(2, xs, 27000, 1, 0.2, 0.15)
		ys += np.random.normal(0, 0.05, len(ys))
		ys += 1.1
		
		mainwindow.mainwidget.ax.set_xlim(*xrange)
		
		mainwindow.mainwidget.freqs = xs
		mainwindow.mainwidget.xs = ys
		mainwindow.mainwidget.ys = ys
		mainwindow.mainwidget.newdata_available.emit()
	
	def show_fit_file(self):
		notify_info.emit(f'The file holding the results of the lineprofile fits is located under:<br>\'{llwpfile(".fit")}\'')

class QSpinBox(QSpinBox):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# AdaptiveDecimalStepType is not implemented in earlier versions of PyQt5
		try:
			self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
		except:
			pass

	def setSingleStep(self, value):
		self.setStepType(QAbstractSpinBox.StepType.DefaultStepType)
		super().setSingleStep(value)

	def setValue(self, value):
		if value < -2147483647 or value > 2147483647:
			value = 0
		return super().setValue(value)

	def setRange(self, min, max):
		min = min if not min is None else -2147483647
		max = max if not max is None else +2147483647
		return super().setRange(min, max)

class QDoubleSpinBox(QDoubleSpinBox):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setDecimals(20)
		# AdaptiveDecimalStepType is not implemented in earlier versions of PyQt5
		try:
			self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
		except:
			pass

	def setSingleStep(self, value):
		self.setStepType(QAbstractSpinBox.StepType.DefaultStepType)
		super().setSingleStep(value)

	def textFromValue(self, value):
		if value and abs(np.log10(abs(value))) > 5:
			return(f"{value:.2e}")
		else:
			return(f"{value:.10f}".rstrip("0").rstrip("."))

	def valueFromText(self, text):
		return(np.float64(text))

	def setRange(self, min, max):
		min = min if not min is None else -np.inf
		max = max if not max is None else +np.inf
		return super().setRange(min, max)

	def validate(self, text, position):
		try:
			np.float64(text)
			return(QValidator.State(2), text, position)
		except ValueError:
			if text.strip() in ["+", "-", ""]:
				return(QValidator.State(1), text, position)
			elif re.match(r"^[+-]?\d+\.?\d*[Ee][+-]?\d?$", text):
				return(QValidator.State(1), text, position)
			else:
				return(QValidator.State(0), text, position)

	def fixup(self, text):
		tmp = re.search(r"[+-]?\d+\.?\d*", text)
		if tmp:
			return(tmp[0])
		else:
			return(str(0))

class FigureCanvas(FigureCanvas):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.wheelEvent = lambda event: event.ignore()
		# self.setStyleSheet('background-color: #00000000')

		app = QApplication.instance()
		app.styleHints().colorSchemeChanged.connect(self.update_theme)
		self.update_theme()

	def update_theme(self):
		background = 'black' if is_dark_theme() else 'white'
		textcolor = 'white' if is_dark_theme() else 'black'
		
		figure = self.figure

		for ax in figure.get_axes():
			ax.tick_params(color=textcolor, labelcolor=textcolor)
			for spine in ax.spines.values():
				spine.set_edgecolor(textcolor)

		self.setStyleSheet(f'background-color: {background}')
		self.draw_idle()

class GUIAbortedError(Exception):
	pass

class Color(str):
	@staticmethod
	def trgb_to_rgbt(color):
		if len(color) == 9:
			color = f"#{color[3:]}{color[1:3]}"
		return(color)

	@staticmethod
	def rgbt_to_trgb(color):
		if len(color) == 9:
			color = f"#{color[-2:]}{color[1:-2]}"
		return(color)

	def __new__(cls, color):
		cls.validate_color(cls, color)
		return super().__new__(cls, color)

	def __assign__(self, color):
		self.validate_color(color)
		return super().__new__(color)

	def validate_color(self, color):
		match = re.search(r'^#(?:[0-9a-fA-F]{3}?){1,2}$|^#(?:[0-9a-fA-F]{8}?)$', color)
		if match:
			if len(color) == 9 and color[-2:] == "ff":
				color = color[:-2]
			return(color)
		else:
			raise CustomError(f"Invalid Color: '{color}' is not a valid color.")

class Geometry():
	_data = {}
	
	@classmethod
	def set(cls, key, value):
		cls._data[key] = value
	
	@classmethod
	def get(cls, key, default=None):
		return(cls._data.get(key, default))

	@classmethod
	def load(cls):
		filename = llwpfile('.geometry')
		if not os.path.isfile(filename):
			return
		
		with open(filename, 'rb') as file:
			cls._data = pickle.load(file)
	
	@classmethod
	def save(cls):
		filename = llwpfile('.geometry')
		with open(filename, 'wb') as file:
			pickle.dump(cls._data, file)

	@classmethod
	def save_widget_geometry(cls, widget):
		key = widget.__class__.__name__
		geometry = widget.geometry()
		cls.set(key, geometry)
	
	@classmethod
	def load_widget_geometry(cls, widget):
		key = widget.__class__.__name__
		geometry = cls.get(key)
		if geometry:
			widget.setGeometry(geometry)

class Config(dict):
	initial_values = {
		'address_synthesizer': ('GPIB::19', str),
		'address_lockin': ('GPIB::12', str),
		'address_pressuregauge': ('COM1', str),

		'measurement_center': (23870.1296, float),
		'measurement_span': (10, float),
		'measurement_stepsize': (50, float),
		'measurement_repetitions': (1, int),
		'measurement_timeconstant': ('20ms', str),
		'measurement_modulationtype': ('2f-FM', str),
		'measurement_modulationfrequency': (10, float),
		'measurement_modulationdeviation': (600, float),
		'measurement_lockinsensitivity': ('20µV', str),
		'measurement_additionaldelaytime': (5e-3, float),
		'measurement_acgain': (4, float),
		'measurement_notes': ('', str),
		'measurement_skipinitialization': (False, bool),
		'measurement_skippressure': (False, bool),
		'measurement_frequencymultiplication': (2, int),

		'plot_dpi': (100, int),
		'plot_ymargin': (0.1, float),
		'plot_ycomponent': ('X', str),

		'color_exp': ('#ffffff', Color),
		'color_fit': ('#fe6100', Color),

		'flag_matplotlibtoolbar': (False, bool),
		'flag_xformatfloat': (".4f", str),
		'flag_logmaxrows': (1000, int),
		'flag_statusbarmaxcharacters': (100, int),
		'flag_notificationtime': (2000, int),
		'flag_pressurefontsize': (40, float),
		'flag_autophase': (True, float),

		'fit_xpoints': (1000, int),
		'fit_fitmethod': ('Voigt 2nd Derivative', str),
		'fit_copytoclipboard': (True, bool),
		'fit_peakdirection': (1, int),
		'fit_polynomrank': (2, int),
		'fit_polynommaxrank': (10, int),
		'fit_offset': (True, bool),
	}

	def __init__(self, signal, *args, **kwargs):
		super().__init__({key: value[0] for key, value in self.initial_values.items()}, *args, **kwargs)
		self._group_callbacks_counter = AtomicCounter()
		self._grouped_callbacks = set()
		self._grouped_callbacks_lock = threading.Lock()
		self.valuechanged = signal
		self.valuechanged.connect(self.callback)
		self.callbacks = pd.DataFrame(columns=["id", "key", "widget", "function"], dtype="object").astype({"id": np.uint})

	def __setitem__(self, key, value, widget=None):
		if self.get(key) != value:
			super().__setitem__(key, value)
			self.valuechanged.emit((key, value, widget))

	def callback(self, args):
		key, value, widget = args
		if widget:
			callbacks = self.callbacks.query(f"key == @key and widget != @widget")["function"].values
		else:
			callbacks = self.callbacks.query(f"key == @key")["function"].values
		
		counter_value = self._group_callbacks_counter.get_value()
		if counter_value:
			with self._grouped_callbacks_lock:
				self._grouped_callbacks.update(callbacks)
		else:
			for callback in callbacks:
				callback()

	def register(self, keys, function):
		if not isinstance(keys, (tuple, list)):
			keys = [keys]
		for key in keys:
			# id is only needed for callback tied to widgets
			id = 0
			df = self.callbacks
			df.loc[len(df), ["id", "key", "function"]] = id, key, function

	def register_widget(self, key, widget, function):
		ids = set(self.callbacks["id"])
		id = 1
		while id in ids:
			id += 1
		df = self.callbacks
		df.loc[len(df), ["id", "key", "function", "widget"]] = id, key, function, widget
		widget.destroyed.connect(lambda x, id=id: self.unregister_widget(id))

	def unregister_widget(self, id):
		self.callbacks.drop(self.callbacks[self.callbacks["id"] == id].index, inplace=True)

	def load(self):
		fname = llwpfile(".ini")
		config_parser = configparser.ConfigParser(interpolation=None)
		config_parser.read(fname)

		self.messages = []
		for section in config_parser.sections():
			for key, value in config_parser.items(section):
				fullkey = f"{section.lower()}_{key.lower()}"
				if fullkey in self.initial_values:
					try:
						class_ = self.initial_values[fullkey][1]
						if class_ in (dict, list, tuple):
							value = json.loads(value)
						elif class_ == bool:
							value = True if value in ["True", "1"] else False
						elif class_ == str:
							value = value.encode("utf-8").decode("unicode_escape")
						value = class_(value)
						self[fullkey] = value
					except Exception as E:
						message = f"The value for the option {fullkey} from the option file was not understood."
						self.messages.append(message)
						print(message)
				else:
					self[fullkey] = value
		
		# Special case changing colors for better contrast
		for key, value in self.items():
			if key in self.initial_values and self.initial_values[key][1] == Color:
				if is_dark_theme():
					if matplotlib.colors.to_hex(value) == "#000000":
						self[key] = "#ffffff"
						self.messages.append(f"Changed the color of '{key}' from black to white as it is otherwise invisible.")
				else:
					if matplotlib.colors.to_hex(value) == "#ffffff":
						self[key] = "#000000"
						self.messages.append(f"Changed the color of '{key}' from white to black as it is otherwise invisible.")
	
	def save(self):
		output_dict = {}
		for key, value in self.items():
			category, name = key.split("_", 1)
			category = category.capitalize()
			if category not in output_dict:
				output_dict[category] = {}
			if type(value) in (dict, list, tuple):
				value = json.dumps(value)
			elif type(value) == str:
				value = value.encode("unicode_escape").decode("utf-8")

			output_dict[category][name] = value

		config_parser = configparser.ConfigParser(interpolation=None)
		for section in output_dict:
			config_parser.add_section(section)
			for key in output_dict[section]:
				config_parser.set(section, key, str(output_dict[section][key]))

		with open(llwpfile(".ini"), "w+", encoding="utf-8") as file:
			config_parser.write(file)
		notify_info.emit("Options were saved successfully!")

	def __enter__(self):
		self._group_callbacks_counter.increase()

	def __exit__(self, type, value, traceback):
		counter_value = self._group_callbacks_counter.decrease()
		if counter_value != 0:
			return
		
		with self._grouped_callbacks_lock:
			for callback in self._grouped_callbacks:
				callback()
			self._grouped_callbacks = set()

class StatusBar(QStatusBar):
	set_cursor_text = pyqtSignal(str)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self._disappearing_messages = []
		self._disappearing_messages_timer = QTimer()
		self._disappearing_messages_timer.setSingleShot(True)
		self._disappearing_messages_timer.timeout.connect(self.next_disappearing_message)

		self.setStyleSheet("QStatusBar::item {border-left: 1px solid inherit;}")

		self.messages_label = QQ(QLabel, text='', wordwrap=True)
		self.addPermanentWidget(self.messages_label, 1)

		self.position_label = QQ(QLabel, text='')
		self.addPermanentWidget(self.position_label)

		self.working_label = QQ(QLabel, text='')
		self.addPermanentWidget(self.working_label)
		
		mainwindow.working.connect(lambda: self.working_label.setText("   Working ...  "))
		mainwindow.waiting.connect(lambda: self.working_label.setText("   Ready  "))

		notify_error.connect(lambda x: self.disappearing_message(x, style='error'))
		notify_warning.connect(lambda x: self.disappearing_message(x, style='warning'))
		notify_info.connect(lambda x: self.disappearing_message(x, style='info'))

	def disappearing_message(self, text, style=None):
		style_string = {
			'error': "<span style='color:#ff0000;'>ERROR</span>: ",
			'warning': "<span style='color:#eda711;'>WARNING</span>: ",
			'info': "<span style='color:#0096FF;'>INFO</span>: ",
		}.get(style, '')

		max_characters = config['flag_statusbarmaxcharacters']
		
		if len(text) > max_characters:
			text = f'{text[:max_characters]} ...'
		
		text = f'  {style_string}{text}'
		self._disappearing_messages.append(text)

		if not self._disappearing_messages_timer.isActive():
			self.next_disappearing_message()

	def next_disappearing_message(self):
		try:
			if len(self._disappearing_messages) > 3:
				self._disappearing_messages = []
				next_message = 'Many new messages. Please see the log window for all messages.'
			else:
				next_message = self._disappearing_messages.pop(0)
		except IndexError:
			self.messages_label.setText('')
			return
		
		self.messages_label.setText(next_message)
		self._disappearing_messages_timer.start(config['flag_notificationtime'])

def QQ(widgetclass, config_key=None, **kwargs):
	widget = widgetclass()

	if "range" in kwargs:
		widget.setRange(*kwargs["range"])
	if "maxWidth" in kwargs:
		widget.setMaximumWidth(kwargs["maxWidth"])
	if "maxHeight" in kwargs:
		widget.setMaximumHeight(kwargs["maxHeight"])
	if "minWidth" in kwargs:
		widget.setMinimumWidth(kwargs["minWidth"])
	if "minHeight" in kwargs:
		widget.setMinimumHeight(kwargs["minHeight"])
	if "color" in kwargs:
		widget.setColor(kwargs["color"])
	if "text" in kwargs:
		widget.setText(kwargs["text"])
	if "options" in kwargs:
		options = kwargs["options"]
		if isinstance(options, dict):
			for key, value in options.items():
				widget.addItem(key, value)
		else:
			for option in kwargs["options"]:
				widget.addItem(option)
	if "width" in kwargs:
		widget.setFixedWidth(kwargs["width"])
	if "height" in kwargs:
		widget.setFixedHeight(kwargs["height"])
	if "tooltip" in kwargs:
		widget.setToolTip(kwargs["tooltip"])
	if "placeholder" in kwargs:
		widget.setPlaceholderText(kwargs["placeholder"])
	if "singlestep" in kwargs:
		widget.setSingleStep(kwargs["singlestep"])
	if "wordwrap" in kwargs:
		widget.setWordWrap(kwargs["wordwrap"])
	if "align" in kwargs:
		widget.setAlignment(kwargs["align"])
	if "rowCount" in kwargs:
		widget.setRowCount(kwargs["rowCount"])
	if "columnCount" in kwargs:
		widget.setColumnCount(kwargs["columnCount"])
	if "move" in kwargs:
		widget.move(*kwargs["move"])
	if "default" in kwargs:
		widget.setDefault(kwargs["default"])
	if "textFormat" in kwargs:
		widget.setTextFormat(kwargs["textFormat"])
	if "checkable" in kwargs:
		widget.setCheckable(kwargs["checkable"])
	if "shortcut" in kwargs:
		widget.setShortcut(kwargs["shortcut"])
	if "parent" in kwargs:
		widget.setParent(kwargs["parent"])
	if "completer" in kwargs:
		widget.setCompleter(kwargs["completer"])
	if "hidden" in kwargs:
		widget.setHidden(kwargs["hidden"])
	if "visible" in kwargs:
		widget.setVisible(kwargs["visible"])
	if "stylesheet" in kwargs:
		widget.setStyleSheet(kwargs["stylesheet"])
	if "enabled" in kwargs:
		widget.setEnabled(kwargs["enabled"])
	if "items" in kwargs:
		for item in kwargs["items"]:
			widget.addItem(item)
	if "readonly" in kwargs:
		widget.setReadOnly(kwargs["readonly"])
	if "prefix" in kwargs:
		widget.setPrefix(kwargs["prefix"])
	if kwargs.get("buttons", True) is False:
		widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

	if widgetclass in [QSpinBox, QDoubleSpinBox]:
		setter = widget.setValue
		changer = widget.valueChanged.connect
		getter = widget.value
	elif widgetclass == QCheckBox:
		setter = widget.setChecked
		changer = widget.stateChanged.connect
		getter = widget.isChecked
	elif widgetclass == QPlainTextEdit:
		setter = widget.setPlainText
		changer = widget.textChanged.connect
		getter = widget.toPlainText
	elif widgetclass == QLineEdit:
		setter = widget.setText
		changer = widget.textChanged.connect
		getter = widget.text
	elif widgetclass == QAction:
		setter = widget.setChecked
		changer = widget.triggered.connect
		getter = widget.isChecked
	elif widgetclass == QPushButton:
		setter = widget.setDefault
		changer = widget.clicked.connect
		getter = widget.isDefault
	elif widgetclass == QToolButton:
		setter = widget.setChecked
		changer = widget.clicked.connect
		getter = widget.isChecked
	elif widgetclass == QComboBox:
		setter = widget.setCurrentText
		changer = widget.currentTextChanged.connect
		getter = widget.currentText
	else:
		return widget

	if "value" in kwargs:
		setter(kwargs["value"])
	if config_key:
		setter(config[config_key])
		changer(lambda x=None, key=config_key: config.__setitem__(key, getter(), widget))
		config.register_widget(config_key, widget, lambda: setter(config[config_key]))
	if "change" in kwargs:
		changer(kwargs["change"])
	if "changes" in kwargs:
		for change in kwargs["changes"]:
			changer(change)

	return widget

def except_hook(cls, exception, traceback):
	if isinstance(exception, GUIAbortedError):
		return
	
	if issubclass(cls, KeyboardInterrupt):
		sys.exit(0)

	sys.__excepthook__(cls, exception, traceback)
	with open(llwpfile(".err"), "a+", encoding="utf-8") as file:
		time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
		file.write(f"{time_str}: \n{exception}\n{''.join(tb.format_tb(traceback))}\n\n")
	try:
		notify_error.emit(f"{exception}\n{''.join(tb.format_tb(traceback))}")
	except Exception as E:
		pass

class EQDockWidget(QDockWidget):
	default_position = 2
	default_visible = False

	def __init__(self, *args, **kwargs):
		super().__init__(mainwindow, *args, **kwargs)
		Geometry.load_widget_geometry(self)
		self.is_instantiated = True
		self.show()
		self.activateWindow()
		self.raise_()
		self.setObjectName(self.__class__.__name__)

		if self.default_position is None:
			self.setFloating(True)
		else:
			mainwindow.addDockWidget(Qt.DockWidgetArea(self.default_position), self)
		self.setVisible(self.default_visible)
		
		tmp = QShortcut("Esc", self)
		tmp.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
		tmp.activated.connect(self.close)
		self.__class__.instance = self

	def moveEvent(self, *args, **kwargs):
		Geometry.save_widget_geometry(self)
		return super().moveEvent(*args, **kwargs)

	def resizeEvent(self, *args, **kwargs):
		Geometry.save_widget_geometry(self)
		return super().resizeEvent(*args, **kwargs)

	def show(self, *args, **kwargs):
		screen_box = self.screen().geometry()
		widget_top_left = self.geometry().topLeft()
		widget_bottom_right = self.geometry().bottomRight()
		
		if not (screen_box.contains(widget_top_left) and screen_box.contains(widget_bottom_right)):
			primary_screen = QApplication.instance().primaryScreen()
			self.move(primary_screen.geometry().center()- self.rect().center())
		
		return(super().show(*args, **kwargs))

class ConfigWindow(EQDockWidget):
	default_visible = False
	default_position = None
	available_in = ['LLWP', 'ASAP']

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setWindowTitle('Config')

		vbox = QVBoxLayout()
		scrollarea = QScrollArea()
		widget = QWidget()
		layout = QGridLayout()

		self.updating = True

		scrollarea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
		scrollarea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		scrollarea.setWidgetResizable(True)

		tmp_layout = QHBoxLayout()
		tmp_layout.addWidget(QQ(QPushButton, text='Save as default', change=lambda: config.save()))
		completer = QCompleter(config.keys())
		completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
		tmp_layout.addWidget(QQ(QLineEdit, placeholder="Search", completer=completer, change=lambda x: self.search(x)))
		tmp_layout.addStretch(1)

		vbox.addLayout(tmp_layout)
		self.widgets = {}

		i = 1
		for key, value in config.items():
			text = json.dumps(value) if isinstance(value, (dict, list, tuple)) else str(value)
			tmp_input = QQ(QLineEdit, value=text, change=lambda text, key=key: self.set_value(key, text))
			tmp_oklab = QQ(QLabel, text="Good")
			tmp_label = QQ(QLabel, text=key)

			self.widgets[key] = (tmp_input, tmp_oklab, tmp_label)
			layout.addWidget(tmp_label, i+1, 0)
			layout.addWidget(tmp_input, i+1, 1)
			layout.addWidget(tmp_oklab, i+1, 2)
			i += 1

		layout.setRowStretch(i+1, 1)

		widget.setLayout(layout)
		scrollarea.setWidget(widget)
		vbox.addWidget(scrollarea)

		mainwidget = QWidget()
		self.setWidget(mainwidget)
		mainwidget.setLayout(vbox)

		self.updating = False
		self.visibilityChanged.connect(self.on_visibility_change)

	def on_visibility_change(self, is_visible):
		if is_visible:
			self.timer = QTimer(self)
			self.timer.timeout.connect(self.get_values)
			self.timer.start(200)
		else:
			self.timer.stop()

	def search(self, text):
		for key, value in self.widgets.items():
			if text.lower() in key or text.lower() in value[0].text():
				hidden = False
			else:
				hidden = True
			value[0].setHidden(hidden)
			value[1].setHidden(hidden)
			value[2].setHidden(hidden)

	def get_values(self):
		self.updating = True
		for key, (input, oklabel, label) in self.widgets.items():
			value = config[key]
			if input.hasFocus() or self.widgets[key][1].text() == "Bad":
				continue
			if isinstance(value, (dict, list, tuple)):
				input.setText(json.dumps(value))
			else:
				input.setText(str(value))
		self.updating = False

	def set_value(self, key, value):
		if self.updating:
			return
		converter = Config.initial_values.get(key)
		if converter:
			converter = converter[1]
		input, oklab, label = self.widgets[key]

		try:
			if converter is None:
				pass
			elif converter in (dict, list, tuple):
				value = json.loads(value)
			elif converter == bool:
				value = True if value in ["True", "1"] else False
			else:
				value = converter(value)
			config[key] = value
			oklab.setText("Good")
		except Exception as E:
			oklab.setText("Bad")

class LogWindow(EQDockWidget):
	available_in = ['LLWP', 'ASAP']

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setWindowTitle("Log")

		mainwidget = QGroupBox()
		layout = QVBoxLayout()
		self.setWidget(mainwidget)
		mainwidget.setLayout(layout)

		self.log_area = QTextEdit()
		self.log_area.setReadOnly(True)
		self.log_area.setMinimumHeight(50)

		notify_error.connect(lambda x: self.writelog(x, style='error'))
		notify_warning.connect(lambda x: self.writelog(x, style='warning'))
		notify_info.connect(lambda x: self.writelog(x, style='info'))

		layout.addWidget(self.log_area)

	def writelog(self, text, style=None):
		separator = "<br/>"
		tmp = self.log_area.toHtml()
		tmp = tmp.split(separator)
		if len(tmp)-1 > config["flag_logmaxrows"]:
			self.log_area.setHtml(separator.join(tmp[-config["flag_logmaxrows"]:]))

		time_str = time.strftime("%H:%M", time.localtime())
		

		style_string = {
			'error': "<span style='color:#ff0000;'>ERROR</span>: ",
			'warning': "<span style='color:#eda711;'>WARNING</span>: ",
			'info': "<span style='color:#0096FF;'>INFO</span>: ",
		}.get(style, '')

		text = f"{time_str}: {style_string}{text}{separator}"
		self.log_area.append(text)
		sb = self.log_area.verticalScrollBar()
		sb.setValue(sb.maximum())


pressure_gauge_lock = threading.Lock()
def measure_pressure(address, skip_reading=False, suppress_warning=False):
	if not address or skip_reading:
		return(None)

	try:
		command = 'PRI?\r\n'
		command = command.encode('utf-8')
		
		with pressure_gauge_lock:
			device = serial.Serial(address, **PRESSURE_GAUGE_KWARGS)
			device.write(command)
			time.sleep(0.05)
			response = device.readline()
			response = response.decode('utf-8').strip()	
			device.close()
		
		if not response:
			return(None)
		return(response.replace('PRI=', ''))
		
	except Exception as E:
		if not suppress_warning:
			notify_warning.emit(f'Could not read the pressure. Error reads:\n{E}')
		return(None)

class PressureWindow(EQDockWidget):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setWindowTitle("Pressure")

		self.label = QQ(QTextEdit)
		self.label.setFontPointSize(config['flag_pressurefontsize'])
		self.label.setFontFamily("Courier")
		self.setWidget(self.label)
		self.visibilityChanged.connect(self.on_visibility_change)

		config.register('flag_pressurefontsize', lambda: self.label.setFontPointSize(config['flag_pressurefontsize']))

	def on_visibility_change(self, is_visible):
		if is_visible:
			self.timer = QTimer(self)
			self.timer.timeout.connect(self.update_pressure)
			self.timer.start(200)
		else:
			self.timer.stop()

	def update_pressure(self):
		voltage = measure_pressure(config['address_pressuregauge'], config['measurement_skippressure'], True)
		if voltage is None:
			self.label.setText(f'No Pressure')
		else:
			voltage = voltage.replace('mV', '')
			voltage = float(voltage)/1000
			pressure = 10 ** (voltage - 5.5)

			power_of_ten = np.log10(pressure)
			if power_of_ten < 0:
				pressure *= 1000
				unit = 'ubar'
			elif power_of_ten >= 3:
				pressure /= 1000
				unit = 'bar'
			else:
				unit = 'mbar'

			self.label.setText(f'{pressure:6.2f} {unit}')


def start():
	MeasurementSoftware()

if __name__ == '__main__':
	start()