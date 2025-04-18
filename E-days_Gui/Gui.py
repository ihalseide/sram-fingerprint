from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QColor, QFont, QPixmap
from PyQt5.QtWidgets import QComboBox, QPushButton, QGridLayout, QLabel
import sys
import serial
import time
import dna_main



class MainWindow(QtWidgets.QMainWindow):
    
	def __init__(self, *args, **kwargs):
		super(MainWindow, self).__init__(*args, **kwargs)
		self.setWindowTitle("SRAM DNA")
		self.setAutoFillBackground(True)
		p = self.palette()
		p.setColor(self.backgroundRole(), QColor(1, 1, 1))
		self.setPalette(p)
		self.setGeometry(400, 200, 750, 500)
		self.work_Thread = QThread()
		self.dna = dna_main.DNA_Main()
		self.dna.moveToThread(self.work_Thread)
		self.dna.results_graph.connect(self.Update_Graph)
		self.dna.status.connect(self.Update_Status)
		self.comm_Thread = None
		self.status_message = QLabel(self)
		self.status_message.setText("Status: ")
		self.status_message.setStyleSheet("background-color : white")
		self.status_message.setGeometry(25, 50, 125, 25)
		self.Start_Button()
		self.Stop_Button()


	def Start_Button(self):
		self.start_button = QPushButton("Start", self)
		self.start_button.setFont(QFont("QApplication::font()", 12))
		self.start_button.setGeometry(25, 205, 125, 75)
		self.start_button.setStyleSheet("background-color : green")
		self.start_button.setEnabled(True)
		self.start_button.clicked.connect(self.Start_Click)
  

	def Start_Click(self):
		self.start_button.setEnabled(False)
		self.comm_Thread = dna_main.ThreadClass(None, self.dna.run)
		self.comm_Thread.start()
  
  
	def Stop_Button(self):
		self.stop_button = QPushButton("Stop", self)
		self.stop_button.setFont(QFont("QApplication::font()", 12))
		self.stop_button.setGeometry(25, 285, 125, 75)
		self.stop_button.setStyleSheet("background-color : red")
		self.stop_button.setEnabled(False)
		self.stop_button.clicked.connect(self.Stop_Click)
  

	def Stop_Click(self):
		self.start_button.setEnabled(True)
		self.stop_button.setEnabled(False)
  
  
	def Update_Status(self, bool_status):
		if bool_status == True:
			self.status_message.setStyleSheet("background-color : green")
			self.status_message.setText("Status: Running")
		else:
			self.status_message.setStyleSheet("background-color : red")
			self.status_message.setText("Status: Check chip")


	def Update_Graph(self, pixmap):
		if not hasattr(self, 'graph_label'):
			self.graph_label = QLabel(self)
			self.graph_label.setGeometry(175, 50, 550, 400)  # Adjust position and size as needed
		self.graph_label.setPixmap(pixmap.scaled(self.graph_label.width(), self.graph_label.height(), QtCore.Qt.KeepAspectRatio))


  
  
  
	def Stop_Com(self):
		if self.comm_Thread is not None:
			self.dna.thread_run = False




def main():
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()