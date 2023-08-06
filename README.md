# Diabetes-Predictor-Desktop-Application
This project is designed to predict diabetes based on medical report images using Optical Character Recognition (OCR) and various machine learning models. The system extracts relevant information from the medical report, processes it, and generates a prediction report along with ROC curves to visualize the model performance.

Features

- Optical Character Recognition (OCR) to extract data from medical report images.
- Utilizes a variety of machine learning models for diabetes prediction.
- Generates a prediction report in PDF format with extracted data and prediction outcome.
- Plots Receiver Operating Characteristic (ROC) curves for model evaluation.
- Handles different levels of diabetes prediction: No Diabetes, Mild Diabetes, and Highly Diabetes.
- User-friendly desktop application interface.

Requirements

- Python 3.11
- PyQt5
- OpenCV
- Tesseract-OCR
- PyPDF2
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- ReportLab
- Beautiful Soup
- lxml
  
How to Use

1. Install the required dependencies using the following command:
   pip install -r requirements.txt [Make sure to include the file path for the requirements
2. Run the desktop application:
   python Diabetes Predictor Desktop Application.py [which is located in the folder PCR_Python_Environment]
3. Click on "Upload Medical Report" to select a medical report image (PNG, JPG, JPEG).
4. Click on "Predict" to generate the diabetes prediction report and ROC curves.

Project Structure

- ` Diabetes Predictor Desktop Application.py`: Main script for the PyQt5 desktop application.
- ` Diabetes Predictor with Optical Character Recognition.py`: Contains the machine learning models, data processing, and PDF report generation.
- `ROC_Curves.py`: Generates ROC curves and saves them in a PDF file.
