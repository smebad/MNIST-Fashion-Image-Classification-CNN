# 🧥👖 MNIST Fashion Image Classification 👗👠

## 📝 Project Overview

This project focuses on building a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The dataset consists of grayscale images representing 10 different categories of clothing and accessories, such as T-shirts, trousers, and shoes. Our objective is to preprocess the data, build and train a deep learning model, evaluate its performance, and interpret the results.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation](#installation)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Conclusion and Insights](#conclusion-and-insights)
7. [Technologies Used](#technologies-used)

---

## 📂 Dataset Description

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) contains 70,000 grayscale images, each of 28x28 pixels, representing different categories of fashion items:
- 👗 60,000 images are used for training
- 👖 10,000 images are used for testing

Each image is labeled with a category from 0 to 9, where each number corresponds to a specific item (e.g., T-shirt, bag, shoe). This dataset serves as a more challenging alternative to the classic MNIST handwritten digits dataset.

---

## 💻 Installation

To run this project, you'll need to install the following Python libraries:
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas
- NumPy

You can install these dependencies using the following command:
```bash
pip install tensorflow keras matplotlib seaborn scikit-learn pandas numpy
```
## ⚙️ Methodology
### 1. 🛠️ Data Preprocessing:
  * Normalization: Each pixel value is scaled to the range [0, 1] to improve model training.
  * Reshaping: The data is reshaped to (28, 28, 1) to be compatible with the input shape expected by CNN layers.
  * Label Encoding: The target labels are converted to categorical format for multi-class classification.

### 2. 🏗️ Model Building:
* We designed a Sequential CNN model using the following layers:
  * Convolutional Layers: For feature extraction using 32 and 64 filters.
  * Max Pooling Layers: For down-sampling to reduce spatial dimensions.
  * Flatten Layer: To convert 2D matrix data to a 1D vector for input to the Dense layer.
  * Dense Layers: The output layer has 10 units with a softmax activation function to predict probabilities for each class.
* Dropout layers were added to reduce overfitting.

### 3. 🎓 Model Training and Evaluation:
  * The model was trained using categorical cross-entropy loss and Adam optimizer over 10 epochs with a batch size of 32.
  * We monitored both training and validation accuracy and loss to assess model performance.
  * A confusion matrix and classification report were generated to evaluate precision, recall, and F1-score across all classes.

 ### 4. 📊 Visualizations
  * We visualized training and validation accuracy and loss curves to understand the model's learning behavior over time.
  * The confusion matrix was visualized to identify where the model performed well and where it struggled.
  * Sample predictions were plotted to show individual examples of correct and incorrect classifications. 

## 🎉 Results
The model achieved a good accuracy on the test dataset, indicating it successfully learned to distinguish between different types of fashion items. The classification report and confusion matrix provided insights into the model's performance across each category, highlighting areas for potential improvement.

## 🔍 Conclusion and Insights
In this project, we built and trained a CNN to classify fashion images. Key takeaways include:

  * Data Preprocessing: Proper normalization and reshaping are critical steps to optimize model performance.
  * CNN Model Design: Using convolutional layers with max pooling effectively extracts features from images, and dropout layers help in reducing overfitting.
  * Evaluation Metrics: Confusion matrix and classification reports are valuable tools to understand model performance in multi-class classification tasks.

This project provided practical experience with image classification, demonstrating the potential of CNNs in computer vision tasks. We also gained insights into the importance of preprocessing, model architecture, and evaluation techniques.

## 🛠️ Technologies Used
* TensorFlow & Keras: For building and training the deep learning model.
* Matplotlib & Seaborn: For visualizing accuracy, loss, and confusion matrix.
* Scikit-learn: For model evaluation metrics like confusion matrix and classification report.
* Pandas & NumPy: For data manipulation and handling.
