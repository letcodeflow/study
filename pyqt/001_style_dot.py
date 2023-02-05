import sys

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt
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
        self.drawRandomPoint(paint)
        paint.end()

    def drawRandomPoint(self, paint):
        pen = QPen()
        colors = [Qt.red, Qt.blue, Qt.green]
        size = self.size()
        print(f'높이와 넓이 {size.height()}, {size.width()}')

        for _ in range(1000):
            pen.setColor(QColor(random.choice(colors)))
            pen.setWidth(random.randint(1,20))
            paint.setPen(pen)

            x = random.randint(1,size.width()-1)
            y = random.randint(1,size.height()-1)

            paint.drawPoint(x,y)

app = QApplication(sys.argv)
exc = myapp()
app.exec_()