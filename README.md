Fake News Detection Web App:
This project is a Machine Learning-based Fake News Detection system built using Natural Language Processing (NLP) and deployed as a Flask web application. It allows users to input a news article or text and predicts whether the news is Real or Fake.

🚀 Project Overview:
The model is trained using a dataset of real and fake news articles. Text data is preprocessed using cleaning techniques and transformed into numerical features using TF-IDF Vectorization. A Logistic Regression model is then trained to classify the news.
The application is deployed using Flask and hosted on Render, providing a simple and interactive user interface.

🛠️ Tech Stack:
1.Python
2.Flask
3.Pandas
4.Scikit-learn
5.NumPy
6.HTML & CSS

⚙️ How It Works:
1.User inputs news text through the web interface
2.Text is preprocessed (cleaning, removing noise, etc.)
3.TF-IDF converts text into numerical features
4.Logistic Regression predicts whether news is Fake or Real
5.Result is displayed on the web page

📁 Project Structure:
1.app.py → Flask backend & ML logic
2.templates/index.html → Frontend UI
3.Fake_small.csv, True_small.csv → Reduced dataset
4.requirements.txt → Dependencies
5.runtime.txt → Python version

⚠️ Dataset Note:
The original dataset is large and exceeds GitHub file size limits. Therefore, a reduced (sampled) version of the dataset is used in this project (Fake_small.csv and True_small.csv).
This approach was chosen to:
1.Ensure smooth deployment on Render
2.Avoid GitHub upload limitations
3.Maintain reasonable model performance
4.Enable faster training during app startup

Live Deployment:
The application is live and accessible here.
https://fake-news-detection-project-ahad5.onrender.com

Conclusion:
This project demonstrates an end-to-end ML workflow including data preprocessing, model training, web integration, and deployment. It is designed to be lightweight, practical, and suitable for real-world applications.


