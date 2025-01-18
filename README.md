# Capstone Project

# Sentiment Based Product Recommendation System 

## Overview

This project is a Flask-based application that provides personalized recommendations. 
It includes a machine learning pipeline for sentiment analysis and recommendation generation. 
The app is designed for deployment on platforms like Heroku.

---

## Features

- User-specific recommendations based on a similarity matrix.
- Sentiment analysis on input data using machine learning models.
- Simple, interactive web interface.

---

## Directory Structure
```
Capstone_Project_Sarah/
|
├── app.py                # Main Flask application
├── models/               # Folder for saved models (e.g., .pkl files)
├── src/                  # Source code for additional utilities
│   ├── textsummarize/    # Text summarization submodule 
├── static/               # Static assets (CSS, JS, images)
├── templates/            # HTML templates for Flask
├── sample30.csv          # Example dataset
├── requirements.txt      # Dependencies
├── Procfile              # Heroku deployment file
└── README.md             # Project documentation 
```

---

## Prerequisites

### Install Software:

- Python 3.8+
- Pip package manager
- Git
- Heroku CLI (for deployment)

### Install Dependencies:

Run the following command to install required Python packages:

In bash
pip install -r requirements.txt

---

## Running the Application

1. **Clone the Repository**:
   In bash
   git clone https://github.com/smruthi-23/Capstone_Project_Sarah.git
   cd Capstone_Project_Sarah
   

2. **Activate Virtual Environment**:
   In bash
   python -m venv venv
   On Windows: 
   venv\Scripts\activate
   

3. **Run the Flask App**:
   In bash
   python app.py
  

4. **Access the Application**:
   Open a browser and go to `http://127.0.0.1:5000/`.

---

## Deployment
### Deploy to Heroku
1. **Log in to Heroku**:
   In bash
   heroku login
   

2. **Create a Heroku App**:
   In bash
   heroku create capstone-project-sarah
   

3. **Push the Code to Heroku**:
   In bash
   git push heroku main
   

4. **Open the Deployed App**:
   In bash
   heroku open
   

---

## Key Components
### `app.py`
- Initializes the Flask application.
- Defines routes for the homepage (`/`) and recommendation endpoint (`/recommend`).

### `models/`
- Contains pre-trained models and serialized files (e.g., `random_forest_model.pkl`, `user_similarity.pkl`).

### `src/`
- Includes utilities for text summarization and other project-specific functionality.

### `sample30.csv`
- Example dataset used for training and generating recommendations.

### `requirements.txt`
- Lists all Python dependencies (e.g., Flask, scikit-learn, joblib etc).

---

## Contribution
1. Fork the repository.
2. Create a feature branch:
   In bash
   git checkout -b feature-name
   
3. Commit your changes:
   In bash
   git commit -m "Describe your changes"
   
4. Push to your branch:
   In bash
   git push origin feature-name
   
5. Submit a pull request.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or feedback, feel free to reach out:
- **Email**: [smruthisarah@gmail.com]
- **GitHub**: [https://github.com/smruthi-23]

