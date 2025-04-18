import serial
from PyQt5 import QtCore
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QByteArray
from io import BytesIO
import XNORTest
import test_connection
import save_array


class DNA_Main(QtCore.QObject):
	results_graph = QtCore.pyqtSignal(object) 
	status = QtCore.pyqtSignal(bool)
	stop = QtCore.pyqtSignal()

	def __init__(self):
		super(DNA_Main, self).__init__()
		self.com_port = serial.Serial("COM3")
		self.com_port.baudrate = 115200
		self.com_port.bytesize = 8
		self.com_port.parity = 'N'
		self.com_port.stopbits = 1
		self.com_port.timeout = None
		self.com_port.xonxoff = False
		self.thread_Cleanup_Finished = False
		self.thread_run = True
		self.file = None
		self.array_dump = []
		self.percentages = []
		self.template_names = []

	def run(self):
		if test_connection.main(self) == True:
			self.status.emit(True)
			self.array_dump = save_array.main(self)
            
            # Get the updated template names and percentages
			self.template_names, self.percentages = XNORTest.main(self.array_dump, self.percentages)
			print("made it")
            # Plot the percentages
			plt.figure(figsize=(10, 4))
			plt.plot(self.percentages, marker='o', linestyle='-', color='navy')
			plt.title("Bit Accuracy per Bit Position")
			plt.xlabel("Bit Index")
			plt.ylabel("Percentage of Correct Bits")
			plt.ylim([0, 100])
			plt.grid(True)

            # Save to a buffer instead of a file
			buf = BytesIO()
			plt.savefig(buf, format='png')
			buf.seek(0)
			plt.close()

            # Convert to QPixmap and emit
			qpix = QPixmap()
			qpix.loadFromData(buf.getvalue(), "PNG")
			self.results_graph.emit(qpix)
        
		else:
			self.status.emit(False)
			self.stop.emit()



class ThreadClass(QtCore.QThread):

	completed = QtCore.pyqtSignal()

	def __init__(self, parent = None, func = None, *args):
		super(ThreadClass, self).__init__(parent)
		self.is_Running = True
		self.func = func
		self.args = args
  
	def run(self):
		if self.func is not None:
			if len(self.args) > 0:
				self.func(*self.args)
			else:
				self.func()
			self.completed.emit()

	def stop(self):
		self.is_Running = False
		self.terminate()

