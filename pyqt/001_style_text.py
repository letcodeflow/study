import sys

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt

class myapp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.pyqtUI()
        
    def pyqtUI(self):
        self.text = 'hhelo wold towld'
        self.setGeometry(300,300,500,500)
        self.setWindowTitle('QPAinter!')
        self.show() 
        
    def painterEvent(self, event):
        paint = QPainter()
        paint.begin(self)
        self.drawText(event, paint)
        paint.end()

    def drawText(self, event, paint):
        paint.setPen(QColor(255,10,255))
        paint.setFont(QFont('Decorateve',10))
        paint.drawText(130,100, 'hello world')
        paint.drawText(event.rect(), Qt.AlignHCenter, self.text)
        
app = QApplication(sys.argv)
exc = myapp()
app.exec_()