# Agrochemical Toxicity Classifier

A machine learning web app that predicts the **toxicity level** of an agrochemical based on its chemical properties and usage type.

## About the Project

This project was built to help classify agrochemicals into three toxicity categories:

| Label | Meaning          |
| ----- | ---------------- |
| 0     | Low concern      |
| 1     | Moderate concern |
| 2     | High concern     |

The model was trained on the **APisTox dataset** — a collection of agrochemical records from sources like ECOTOX, PPDB, and BPDB — using a scikit-learn pipeline that handles preprocessing and classification automatically.

## Features

- Simple web interface built with Streamlit
- Takes 7 input features: year of introduction, herbicide, fungicide, insecticide, other agrochemical type, data source, and toxicity type
- Returns a predicted toxicity label with a color-coded result
- Displays class probabilities as a bar chart

## Project Structure

```
agroClassifyDeploy/
├── app.py                   # Streamlit web application
├── agrochemical_model.pkl   # Trained sklearn model pipeline
├── apistox_data.csv         # Dataset used for training
├── requirements.txt         # Python dependencies
└── README.md
```

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the app:

   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`

## Dependencies

- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn
- joblib

---

https://agroclassifier-akeem.streamlit.app/

---

##

ML Project done by **Akeem Arikeuyo**
