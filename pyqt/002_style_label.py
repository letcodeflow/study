import sys

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap

class myapp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        
        licat = QLabel()
        pie = QLabel()
        sun = QLabel()
        
        licat.setStyleSheet(
            'border-style:solid;'
            'border-wdith:3px;'
            'border-color:red;'
            'border-radius:3px;'
            'image:url(img/weniv-licat.png)'
            )
        pie.setStyleSheet(
            'border-style:double;'
            'border-wdith:3px;'
            'border-color:blue;'
            'border-radius:3px;'
            'image:url(img/weniv-licat.png)'
            )
        sun.setStyleSheet(
            'border-style:dot-dot-dash;'
            'border-wdith:3px;'
            'border-color:red;'
            'border-radius:3px;'
            'background-color:black;'
            'image:url(img/weniv-licat.png)'
            )
        
        hbox = QHBoxLayout()
        hbox.addWidget(licat)
        hbox.addWidget(pie)
        hbox.addWidget(sun)
        
        self.setLayout(hbox)
        
        self.setGeometry(300,300,500,500)
        self.setWindowTitle('QPAinter!')
        self.show() 

        
        
        
app = QApplication(sys.argv)
exc = myapp()
app.exec_()