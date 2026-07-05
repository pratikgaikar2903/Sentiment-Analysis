<img width="1024" height="506" alt="image" src="https://github.com/user-attachments/assets/313a887e-563c-47a8-8be2-75435c267f7b" />


# Emotion Classification Notebook

This notebook demonstrates the process of building and evaluating an emotion classification model using text data.

## Project Overview

The goal of this project is to classify emotions (e.g., joy, anger, sadness) from textual input. It covers data loading, preprocessing, model training, evaluation, and saving the trained components for later use.

## Key Steps:

1.  **Data Loading and Preprocessing**: The notebook loads a dataset containing sentences and their corresponding emotion labels. It handles missing values and duplicate records to ensure data quality.
2.  **Feature Engineering**: Text data is transformed into numerical features using `CountVectorizer`, which converts text into a matrix of token counts.
3.  **Model Training**: A `Multinomial Naive Bayes` classifier is trained on the vectorized text data.
4.  **Model Evaluation**: The model's performance is assessed using various metrics, including precision, recall, F1-score, and accuracy, through a classification report.
5.  **Prediction on Custom Input**: The trained model is tested with a custom sentence to demonstrate its real-world applicability.

## Saved Models

Two essential components of this project have been saved as `.pkl` files:

*   `emotion_model.pkl`: This file contains the trained `Multinomial Naive Bayes` model. It can be loaded to make predictions on new data without needing to retrain the model.
*   `vectorizer.pkl`: This file contains the fitted `CountVectorizer`. It must be loaded and used to transform any new text data into the same numerical format that the model was trained on, ensuring consistent input for predictions.

## How to Use the Saved Models:

To use the saved model and vectorizer, you can load them using `joblib`:

```python
import joblib

# Load the trained model
loaded_model = joblib.load('emotion_model.pkl')

# Load the fitted vectorizer
loaded_vectorizer = joblib.load('vectorizer.pkl')

# Example prediction with a new sentence
new_sentence = ["I am so happy today!"]
processed_sentence = loaded_vectorizer.transform(new_sentence)
prediction = loaded_model.predict(processed_sentence)

print(f"The predicted emotion is: {prediction[0]}")
```
## 👤 Author

<table>
  <tr>
    <td>
      <img src="https://github.com/pratikgaikar2903.png" width="100" style="border-radius: 50%;" alt="Pratik Gaikar"/>
    </td>
    <td>
      <strong>Pratik Gaikar</strong><br>
      Data Scientist / Machine Learning Engineer<br>
      🌐 <a href="https://github.com/pratikgaikar2903">GitHub</a> | 💼 <a href="https://www.linkedin.com/in/pratik-gaikar-5a6431245">LinkedIn</a>
    </td>
  </tr>
</table>
