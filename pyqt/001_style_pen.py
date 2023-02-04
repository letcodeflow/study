import sys

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt

class myapp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.pyqtUI()
        
    def pyqtUI(self):
        self.setGeometry(300,300,500,500)
        self.setWindowTitle('QPAinter!')
        self.show() 
        
    def painterEvent(self, event):
        paint = QPainter()
        paint.begin(self)
        self.drawLine(paint)
        paint.end()

    def drawLine(self, paint):
        pen = QPen(Qt.blue, 4, Qt.SolidLine)
        paint.setPen(pen)
        paint.drawLine(100,40,400,40)
        
        pen.setStyle(Qt.DashLine)
        pen.setColor(Qt.yellow)
        paint.setPen(pen)
        paint.drawLine(100,80,400,80)

        pen.setStyle(Qt.DashDotDotLine)
        pen.setColor(Qt.darkGreen)
        paint.setPen(pen)
        paint.drawLine(100,150,400,160)

        pen.setStyle(Qt.CustomDashLine)
        pen.setDashPattern([1,4,5,4])
        pen.setColor(Qt.darkGreen)
        pen.setWidth(8)
        paint.setPen(pen)
        paint.drawLine(100,200,300,200)
        
app = QApplication(sys.argv)
exc = myapp()
app.exec_()