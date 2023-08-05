#Kindly update the path of the python file ( filename) in line no. 4 and the predictor file ( file name) as the new location saved in your device at all places where file paths have been specified and also kindly make sure that every '\' is replaced with '\\' wherever shown in code.
import sys
path = "C:\\Users\\Hp\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages"
sys.path.append(path)
import os
import PyPDF2
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QFont
from bs4 import BeautifulSoup

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Python Script Runner')

        # Create a vertical layout
        layout = QVBoxLayout()

        # Create a label to display HTML content
        label = QLabel()
        label.setStyleSheet('QLabel { font-size: 18px; padding: 10px; }')
        layout.addWidget(label)

        # Set the layout for the window
        self.setLayout(layout)

        # Load and parse the HTML file
        with open(rf'C:\Users\Hp\Desktop\Predictor\code.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'lxml')

        # Extract the styles from the HTML file and apply them to the PyQt5 application
        style_tag = soup.find('style')
        if style_tag:
            style = style_tag.string
            self.setStyleSheet(style)

        # Extract the data from the HTML file and display it in the label
        data = soup.find('body').get_text()
        label.setText(data)

        # Create a button for image selection
        select_button = QPushButton('Upload Medical Report')
        select_button.setStyleSheet('QPushButton { background-color: #0173b1; color: white; font-size: 18px; padding: 10px; }')
        select_button.clicked.connect(self.selectImage)
        layout.addWidget(select_button)

        
         # Create a button and apply styling
        button = QPushButton('Predict')
        button.setStyleSheet('QPushButton { background-color: #0173b1; color: white; font-size: 18px; padding: 10px; }')
        button.clicked.connect(self.runScript)
        layout.addWidget(button)

    def selectImage(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        if image_path:
            self.image_path = image_path.replace('/', '\\')
            print(f'Selected Image: {self.image_path}')

    def runScript(self):
        script_path = rf'C:\Users\Hp\Desktop\Predictor\OCR_Python_Environment\Diabetes Predictor with Optical Character Recognition.py'
        image_path = self.image_path.replace('/', '\\') if hasattr(self, 'image_path') else ''
        process = QProcess()
        process.start('python', [script_path, self.image_path])

        if not process.waitForStarted():
            print('Failed to start process.')

        process.waitForFinished()

        script_path = rf"C:\Users\Hp\Desktop\Predictor\OCR_Python_Environment\ROC_Curves.py"
        image_path = self.image_path.replace('/', '\\') if hasattr(self, 'image_path') else ''
        process = QProcess()
        process.start('python', [script_path, self.image_path])

        if not process.waitForStarted():
            print('Failed to start process.')

        process.waitForFinished()

        # Retrieve the output of the Python script
        output = process.readAllStandardOutput()
        output_str = str(output, 'utf-8')
        print(output_str)

        input_files = ['C:\\Users\\Hp\\Desktop\\Predictor\\Generated_Report.pdf', 'C:\\Users\\Hp\\Desktop\\Predictor\\OCR_Python_Environment\\Confusion_Matrices.pdf', 'C:\\Users\\Hp\\Desktop\\Predictor\\OCR_Python_Environment\\ROC_Curves.pdf' ]
        output_file = 'C:\\Users\\Hp\\Desktop\\Predictor\\OCR_Python_Environment\\Predictor_Output.pdf'
        
        merger = PyPDF2.PdfMerger()

        for file in input_files:
            with open(file, 'rb') as f:
                merger.append(f)

        with open(output_file, 'wb') as f:
            merger.write(f)

        print(f"Merged PDFs saved as '{output_file}'.")
        
        pdf_file = os.path.join(rf'C:\Users\Hp\Desktop\Predictor\OCR_Python_Environment', 'Predictor_Output.pdf')
        if os.path.isfile(pdf_file):
            if sys.platform == 'darwin':  # macOS
                subprocess.call(('open', pdf_file))
            elif sys.platform == 'win32':  # Windows
                os.startfile(pdf_file)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('QWidget { background-color: #f0f0f0; }')  # Set background color for the window
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())