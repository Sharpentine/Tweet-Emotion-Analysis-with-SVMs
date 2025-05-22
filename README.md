# Emotion Classification with SVM

This project focuses on classifying emotions from text data using Support Vector Machine (SVM) models with different kernels. The process involves loading and cleaning tweet data, transforming the text into numerical features using TF-IDF, splitting the data into training and testing sets, training SVM models, and evaluating their performance.

## Project Structure

The project is organized as follows:

1.  **Data Loading:** Reads the `emotions.csv` dataset and samples a limited number of tweets per emotion.
2.  **Data Cleaning:** Preprocesses the tweet text by handling missing values, removing URLs, mentions, hashtags, and punctuation, removing stopwords, and converting text to lowercase.
3.  **Data Wrangling:** Converts the cleaned text data into numerical features using TF-IDF vectorization.
4.  **Data Splitting:** Divides the data into training and testing sets for model training and evaluation.
5.  **Model Training:** Trains multiple SVM models with different kernels (linear, rbf, poly, sigmoid).
6.  **Model Evaluation:** Evaluates the performance of each trained model using accuracy, precision, recall, and F1-score.
7.  **Model Saving:** Saves the best-performing model (linear kernel in this case) to a file.

## Dependencies

The project requires the following libraries:

*   pandas
*   nltk
*   scikit-learn
*   joblib

You can install the required libraries using pip:
bash pip install pandas nltk scikit-learn joblib
You will also need to download the `punkt` and `punkt_tab` resources from nltk:
python import nltk 
nltk.download('punkt') 
nltk.download('punkt_tab')
## Usage

1.  Make sure you have the `emotions.csv` file in the same directory as your notebook or script.
2.  Run the code cells in order to execute the data loading, cleaning, wrangling, splitting, model training, and evaluation steps.
3.  The evaluation results for each kernel will be printed, and the best-performing model will be saved as `best_svm_model.joblib`.

## Dataset

The project uses the "emotions.csv" dataset, which contains tweet text and corresponding emotion labels.

## Results

The evaluation results for each SVM kernel are printed to the console. Based on the initial evaluation, the linear kernel showed the best performance and was selected for saving. Further hyperparameter tuning could be performed on the best-performing model to potentially improve results.

## Model Saving

The best-performing SVM model (with the linear kernel) is saved to the file `best_svm_model.joblib` using the `joblib` library. This saved model can be loaded later for making predictions on new data.
