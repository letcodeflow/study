import sys

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPolygon
from PyQt5.QtCore import Qt, QPoint
import random

class myapp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.pyqtUI()
        
    def pyqtUI(self):
        self.setGeometry(300,300,500,500)
        self.setWindowTitle('QPAinter!')
        self.show() 
        
    def paintEvent(self, event):
        paint = QPainter()
        paint.begin(self)
        self.다각형(paint)
        paint.end()
    
    def 다각형(self, paint):
        점 = [
            QPoint(10,10),
            QPoint(20,30),
            QPoint(130,120),
            QPoint(260,140),
            QPoint(360,160),
            QPoint(220,270),
            QPoint(200,190),
        ]
        
        
        다각형 = QPolygon(점)
        paint.setPen(QPen(Qt.red,10))
        paint.drawPolygon(다각형)

        점 = [
            QPoint(10,10),
            QPoint(360,160),
            QPoint(220,270),
            QPoint(200,190),
        ]
        
        
        다각형_둘 = QPolygon(점)
        paint.setPen(QPen(Qt.blue,10))
        paint.drawPolygon(다각형_둘)
        
        
        #호 현 파이
        #x,y, width, height, starg agle, span angle
        paint.setPen(QPen(Qt.black, 5))
        paint.drawArc(100,300,100,100,0*16, 180*16)
        paint.drawText(150,410,'180')

        paint.setPen(QPen(Qt.red, 5))
        paint.drawChord(250,20,100,100,270*16, 60*16)
        paint.drawText(150,410,'60')

        paint.setPen(QPen(Qt.blue, 5))
        paint.drawPie(250,120,100,100,90*16, 180*16)
        paint.drawText(150,410,'180')

        
        
        
app = QApplication(sys.argv)
exc = myapp()
app.exec_()