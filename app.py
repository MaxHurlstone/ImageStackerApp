
from PyQt6.QtCore import QDateTime, Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFileDialog, QAbstractSlider)
from PyQt6.QtGui import (QPixmap, QImage)
from functools import partial
import imagestacker

class StackingApp(QDialog):
    def __init__(self, parent=None):
        # Automatically create stacking session
        self.stacker = imagestacker.ImageStacker()

        super(StackingApp, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        disableWidgetsCheckBox = QCheckBox("&Disable widgets")

        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createBottomRightGroupBox()
        self.createProgressBar()

        disableWidgetsCheckBox.toggled.connect(self.topLeftGroupBox.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.topRightGroupBox.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.bottomRightGroupBox.setDisabled)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0, 2, 1)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        mainLayout.addWidget(self.progressBar, 3, 0, 1, 2)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.setWindowTitle("Stacking App")


    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) // 100)


    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Group 1")

        # Buttons
        button_rawdir = QPushButton("Select Raw Img Dir")
        button_rawdir.clicked.connect(partial(self.selectDirectoryDialog, 'rawdir'))
        button_savedir = QPushButton("Select Save Img Dir")
        button_savedir.clicked.connect(partial(self.selectDirectoryDialog, 'savedir'))
        button_opnimgs = QPushButton("Load Images!")
        button_opnimgs.clicked.connect(self.openImages)

        # Text to show selected directories
        self.rawdir = QLabel("")
        self.savedir = QLabel("")

        # Drop down for tracking method
        dropdwn_tracking = QComboBox()
        dropdwn_tracking.addItem("Peaks")
        dropdwn_tracking.addItem("Crosscor")
        dropdwn_tracking.addItem("Circle")
        dropdwn_tracking.addItem("Crescent")
        dropdwn_tracking.currentTextChanged.connect(self.selectMethod)

        layout = QVBoxLayout()
        layout.addWidget(button_rawdir)
        layout.addWidget(self.rawdir)
        layout.addWidget(button_savedir)
        layout.addWidget(self.savedir)
        layout.addWidget(button_opnimgs)
        # layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)
        self.topLeftGroupBox.setLayout(layout)


    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Group 2")

        self.label = QLabel(self)
        self.pixmap = QPixmap('cat.jpeg')
        # self.label.setPixmap(self.pixmap)

        self.slider = QSlider(Qt.Orientation.Horizontal, self.topRightGroupBox)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.updateRawImg)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.topRightGroupBox.setLayout(layout)


    def createBottomRightGroupBox(self):

        self.bottomRightGroupBox = QGroupBox("Group 2")

        label = QLabel(self)
        pixmap = QPixmap('cat.jpeg')
        label.setPixmap(pixmap)

        layout = QVBoxLayout()
        layout.addWidget(label)
        self.bottomRightGroupBox.setLayout(layout)


    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

        timer = QTimer(self)
        timer.timeout.connect(self.advanceProgressBar)
        timer.start(1000)


    def selectDirectoryDialog(self, value):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Directory")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)

        if file_dialog.exec():
            selected_directory = file_dialog.selectedFiles()[0]
            print("Selected Directory:", selected_directory)

            if value == 'rawdir':
                self.stacker.data_dir = selected_directory
                self.rawdir.setText(selected_directory)
            elif value == 'savedir':
                self.stacker.save_dir = selected_directory
                self.savedir.setText(selected_directory)


    def selectMethod(self, method):
        if method == "Peaks":
            self.stacker.peaks()
        elif method == "Crosscor":
            self.stacker.crosscor()
        elif method == "Circle":
            self.stacker.circle()
        elif method == "Crescent":
            self.stacker.crescent()

    def openImages(self):
        self.stacker.open_images()
        self.slider.setMaximum(self.stacker.num_images-1)

    def updateRawImg(self, i):
        self.label = QLabel(self)

        img = self.stacker.raw_images[i]

        self.convert = QImage(img, img.shape[1], img.shape[0], QImage.Format.Format_BGR888)
        self.pixmap = QPixmap('cat.jpeg')
        self.label.setPixmap(self.pixmap)
        # self.label.setPixmap(QPixmap.fromImage(self.convert))

        print("Slider moved to:", i)

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = StackingApp()
    gallery.show()
    sys.exit(app.exec())
