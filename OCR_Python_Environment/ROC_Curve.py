import sys
sys.path.append("C:\\Users\\Hp\\Appdata\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc
from scipy.optimize import curve_fit

# Function to fit a quadratic polynomial
def quadratic(x, a, b, c):
    return a * x + b * x**2

def plot_confusion_matrix(confusion_matrix, title, ax):
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


def calculate_roc_curve(confusion_matrix):
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    TN = confusion_matrix[0, 0]
    FN = confusion_matrix[1, 0]

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return TPR, FPR


confusion_matrices = [
    np.array([[98, 9], [18, 29]]),
    np.array([[98, 9], [23, 24]]),
    np.array([[94, 13], [17, 30]]),
    np.array([[87, 20], [16, 31]]),
    np.array([[87, 20], [18, 29]]),
    np.array([[93, 14], [18, 29]]),
    np.array([[75, 32], [27, 20]]),
    np.array([[50, 57], [12, 35]])
]

models = [
    "Logistic Regression Model",
    "SVC Model",
    "Random Forest Model",
    "Decision Tree Model",
    "KNN Model",
    "Gaussian NB Model",
    "MLP Model",
    "Perceptron Model"
]


with PdfPages('C:/Users/Hp/Desktop/Predictor/OCR_Python_Environment/ROC_Curves.pdf') as pdf:
    # Generate individual ROC curves and save them in the PDF
    for i, confusion_matrix in enumerate(confusion_matrices):
        plt.figure(figsize=(8, 8), dpi=100)

        TPR, FPR = calculate_roc_curve(confusion_matrix)

        # Interpolate the ROC curve for more detailed plot with 1000 points
        interp_fprs = np.linspace(0, 1, 10)
        interp_tprs = np.interp(interp_fprs, [FPR, 1], [TPR, 1])

        # Fit a quadratic polynomial to the interpolated TPR and FPR values
        popt, _ = curve_fit(quadratic, interp_fprs, interp_tprs)

        # Calculate the area under the curve (AUC)
        auc_score = auc(interp_fprs, interp_tprs)

        # Generate smooth TPR and FPR values using the fitted polynomial
        smooth_fprs = np.linspace(0, 1, 10)
        smooth_tprs = quadratic(smooth_fprs, *popt)

        # Plot the ROC curve using the smooth TPR and FPR values
        plt.plot(smooth_fprs, smooth_tprs, lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal reference line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ' + models[i] + '\nAUC = {:.3f}'.format(auc_score))
        plt.grid(True)
        plt.axis('equal')  # Set equal intervals on x-axis and y-axis
        plt.tight_layout()

        # Save the individual ROC curve to the PDF
        pdf.savefig()
        plt.close()

    # Generate a combined ROC curve on the second page
    plt.figure(figsize=(8, 8), dpi=100)

    for i, confusion_matrix in enumerate(confusion_matrices):
        TPR, FPR = calculate_roc_curve(confusion_matrix)

        # Interpolate the ROC curve for more detailed plot with 100 points
        interp_fprs = np.linspace(0, 1, 10)
        interp_tprs = np.interp(interp_fprs, [FPR, 1], [TPR, 1])

        # Fit a quadratic polynomial to the interpolated TPR and FPR values
        popt, _ = curve_fit(quadratic, interp_fprs, interp_tprs)

        # Calculate the area under the curve (AUC)
        auc_score = auc(interp_fprs, interp_tprs)

        # Generate smooth TPR and FPR values using the fitted polynomial
        smooth_fprs = np.linspace(0, 1, 10)
        smooth_tprs = quadratic(smooth_fprs, *popt)

        # Plot the ROC curve using the smooth TPR and FPR values
        plt.plot(smooth_fprs, smooth_tprs, lw=2, label=models[i] + ', AUC = {:.3f}'.format(auc_score))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal reference line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Set equal intervals on x-axis and y-axis
    plt.tight_layout()

    # Save the combined ROC curve to the PDF
    pdf.savefig()
    plt.close()

# Show a message when the PDF file is saved successfully
print("ROC curves saved as 'ROC_Curves.pdf'.")