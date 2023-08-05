#Kindly update the path of the python file ( filename) in line no. 4 and the predictor file ( file name) as the new location saved in your device at all places where file paths have been specified and also kindly make sure that every '\' is replaced with '\\' wherever shown in code.
#Link to download Tesseract OCR-wiki: https://github.com/UB-Mannheim/tesseract/wiki
# Importing required libraries
import os
import sys
sys.path.append("C:\\Users\Hp\\Appdata\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
from PyQt5.QtWidgets import QFileDialog
import cv2
import pytesseract
import pandas as pd
import numpy as np
import PyPDF2


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron


# Set the Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Setting up ML models as Base Classes
class BaseModel:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
		

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
		

class SVCModel(BaseModel):
    def __init__(self):
        super().__init__("SVM Classifier")
        self.model = SVC()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
		

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("Random Forest Classifier")
        self.model = RandomForestClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
		

class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__("Decision Tree Classifier")
        self.model = DecisionTreeClassifier()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
		

class KNNModel(BaseModel):
    def __init__(self):
        super().__init__("KNN Classifier")
        self.model = KNeighborsClassifier()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
		

class GaussianNBModel(BaseModel):
    def __init__(self):
        super().__init__("Naive Bayes")
        self.model = GaussianNB()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
		

class MLPModel(BaseModel):
    def __init__(self):
        super().__init__("Neural Network")
        self.model = MLPClassifier(hidden_layer_sizes=(10,))

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
		

class PerceptronModel(BaseModel):
    def __init__(self):
        super().__init__("Perceptron")
        self.model = Perceptron()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
		

#Loading Dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)
	

#Preprocessing Dataset
def preprocess_dataset(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    return X, y
	

#Model Evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    return cm_train, cm_test, accuracy_train, accuracy_test


#OCR Operation
def perform_ocr(image_path):
    input_image = cv2.imread(image_path)
    return pytesseract.image_to_string(input_image)
	

#Data Extraction
def extract_data_from_text(text):
    data = {}
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        split_values = line.split(':', 1)
        if len(split_values) == 2:
            key = split_values[0].strip()
            value = split_values[1].strip()
            try:
                data[key] = float(value) if '.' in value else int(value)
                print(f'{key}: {data[key]}')
            except ValueError:
                print("Invalid value format:", line)
        else:
            continue
    return data
	

#Comparison Graph Plot
def plot_data_comparison(common_keys, X, update_data):
    for column in common_keys:
        plt.figure()
        train_data = X[column]
        extracted_data = update_data[column]
        plt.hist(train_data, bins=10, alpha=0.5, label='Training Data')
        plt.axvline(x=np.mean(train_data), color='g', linestyle='dashed', linewidth=2, label='Average Training Data')
        for val in extracted_data:
            plt.axvline(x=val, color='r', linestyle='dashed', linewidth=2, label='Extracted Data')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Data Comparison: {column}')
        plt.legend()
        plot_image_path = rf'C:\Users\Hp\Desktop\Predictor\OCR_Python_Environment\extracted_data_comparison_{column}.png'
        plt.savefig(plot_image_path)
        plt.close()
		

#Prediction
def prediction_data(data, models):
    X, y = preprocess_dataset(load_dataset(rf'C:\Users\Hp\Desktop\Predictor\OCR_Python_Environment\Diabetes.csv'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # Reshape the data from a scalar array to a 2D array with a single sample
    data = np.array([list(data.values())]).reshape(1, -1)

    predict = []
    for model in models:
        model.fit(X_train, y_train)
        predict.append(model.predict(data)[0])
    print(predict)
    return predict
	

#Prediction Text Generation
def prepare_prediction_text(data, models):
    i = 0
    pos_result = 0
    neg_result = 0
    individual_predictions = []
    prediction_text = ""
    if len(data) == 0:
        prediction_text = "No data fields extracted. Please ensure the OCR extraction is accurate."
    else:
        extracted_keys = list(set(data.keys()))
        if len(extracted_keys) == 0:
            prediction_text = "No valid data fields extracted for prediction."
        else:
            individual_predictions = prediction_data(data, models)
            pos_result = sum(1 for prediction in individual_predictions if prediction == 1)
            neg_result = sum(1 for prediction in individual_predictions if prediction == 0)
            if pos_result > neg_result:
                prediction_text = "The patient has diabetes"
                # Additional check for blood glucose levels
                if "Glucose" in data:
                    glucose_level = data["Glucose"]
                    if glucose_level > 180:
                        prediction_text += " and is highly diabetic."
                    elif glucose_level > 140:
                        prediction_text += " and is mildly diabetic."
            else:
                prediction_text = "The patient does not have diabetes."
    print(prediction_text)
    return prediction_text



#PDF Generation
def generate_pdf_report(data, prediction_text, common_keys, models, predictions):
    i = 0
    output_pdf_path = rf'C:\Users\Hp\Desktop\Predictor\Generated_Report.pdf'
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(200, 750, "Diabetes Prediction Report")
    c.setFont("Helvetica", 16)
    c.drawString(50, 700, "Extracted Data:")
    c.setFont("Helvetica", 10)
    y_position = 670
    for key, value in data.items():
        c.drawString(70, y_position, f"{key}: {value}")
        y_position -= 40
    prediction_text = prepare_prediction_text(data, models)
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position - 20, "Prediction Report:")
    c.setFont("Helvetica", 10)
    c.drawString(70, y_position - 40, prediction_text)
    if len(common_keys) > 0:
        page_number = 2
        
        for column in common_keys:
            c.showPage()
            c.setFont("Helvetica", 12)
            c.drawString(50, 750, f'Data Comparison: {column}')
            c.setFont("Helvetica", 10)
            plot_image_path = rf'C:\Users\Hp\Desktop\Predictor\extracted_data_comparison_{column}.png'
            c.drawImage(plot_image_path, 50, 220, width=500, height=400)
            page_number += 1
    c.showPage()
    c.save()
    if "highly diabetic" in prediction_text:
        input_files = ['C:\\Users\\Hp\\Desktop\\Predictor\\Generated_Report.pdf', 'C:\\Users\\Hp\\Desktop\\Predictor\\High Diabetes.pdf']
        output_file = 'C:\\Users\\Hp\\Desktop\\Predictor\\Generated_Report.pdf'
        
        merger = PyPDF2.PdfMerger()

        for file in input_files:
            with open(file, 'rb') as f:
                merger.append(f)

        with open(output_file, 'wb') as f:
            merger.write(f)
    if "mildly diabetic" in prediction_text:
        input_files = ['C:\\Users\\Hp\\Desktop\\Predictor\\Generated_Report.pdf', 'C:\\Users\\Hp\\Desktop\\Predictor\\Mild Diabetes.pdf']
        output_file = 'C:\\Users\\Hp\\Desktop\\Predictor\\Generated_Report.pdf'
        
        merger = PyPDF2.PdfMerger()

        for file in input_files:
            with open(file, 'rb') as f:
                merger.append(f)

        with open(output_file, 'wb') as f:
            merger.write(f)
    print("Output PDF generated successfully.")


def selectImage(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        if image_path:
            self.image_path = image_path
            print(f'Selected Image: {self.image_path}')
            

def main(image_path):
    print("Received Image Path:", image_path)
    
    # Load the pre-trained model and perform prediction
    data_file = load_dataset(r'C:\Users\Hp\Desktop\Predictor\OCR_Python_Environment\Diabetes.csv')
    X, y = preprocess_dataset(data_file)

	#Defining all the models
    models = [
        LogisticRegressionModel(),
        SVCModel(),
        RandomForestModel(),
        DecisionTreeModel(),
        KNNModel(),
        GaussianNBModel(),
        MLPModel(),
        PerceptronModel()
    ]

    weights = []
    predictions = []
    accuracies = []
    for model in models:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        cm_train, cm_test, accuracy_train, accuracy_test = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(model.name)
        print("Accuracy:", accuracy_test * 100, "%")
        print("\n")
        weights.append(accuracy_test)
        predictions.append((model.name, "No Diabetes" if accuracy_test > 0.5 else "Diabetes"))
        accuracies.append(accuracy_test)

    # Load the image using OpenCV
    text = perform_ocr(image_path)

    # Extract data from the OCR text
    data = extract_data_from_text(text)

    # Prepare the prediction text
    prediction_text = prepare_prediction_text(data, models)

    # Plot the extracted data comparison
    extracted_keys = set(data.keys())
    common_keys = extracted_keys.intersection(X.columns)
    if len(common_keys) > 0:
        update_data = pd.DataFrame([data])
        update_data = update_data[list(common_keys)]
        plot_data_comparison(common_keys, X, update_data)

    # Generate the PDF report
    print(data)
    prediction_data(data, models)
    generate_pdf_report(data, prediction_text, common_keys, models, predictions)
    save_path = os.path.join(os.path.expanduser("~"), "Desktop")  # Save on the desktop
    print(accuracies)
    for model in models:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        cm_train, cm_test, accuracy_train, accuracy_test = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(cm_test)
	

#Setting the Program as Main Module
if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else ""
    main(image_path)