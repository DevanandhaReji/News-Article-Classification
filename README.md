# News-Article-Classification
News Article Classification
# Fake/Real News Classification using NLP and ML

## Project Overview
This project detects whether a news article is real or fake using Natural Language Processing (NLP) and Machine Learning (Logistic Regression).
It was developed as part of the Elevate Labs Internship - Project Phase.

## Dataset Used
- True.csv.xlsx → Real news articles
- Fake.csv.xlsx → Fake news articles

Both datasets were combined and preprocessed for training and testing.

## Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- TF-IDF Vectorization
- Logistic Regression
- NLTK for text preprocessing

## Model Details
- Preprocessing: Lowercasing, stopword removal, stemming
- Vectorization: TF-IDF (Top 5000 features)
- Classifier: Logistic Regression
- Accuracy: approximately 98.9 percent on test data

## How to Use

### Test a Custom News:
```python
text = "The government launches a new digital health mission."
cleaned = preprocess(text)
vector = tfidf.transform([cleaned])
model.predict(vector)
```

## Files Included
| File Name                   | Description                        |
|----------------------------|------------------------------------|
| news_model.pkl             | Trained logistic regression model  |
| vectorizer.pkl             | TF-IDF transformer object          |
| News_Classification_Project_Output.pdf | Project summary output       |
| True.csv.xlsx + Fake.csv.xlsx         | Raw datasets                 |

## Example Output
Input: "The government launches a new health mission"  
Prediction: REAL

## Optional: Streamlit App
To run the Streamlit app (if added):
```bash
streamlit run app.py
```

## Author
Devanandha PR  
Intern at Elevate Labs (AI/ML Track)
