# Team 4

Team Members:
- Ashwin Muthuraman
- Srinivas Saiteja
- Kanishk Goel
- Ram Mannuru

## Research Topic

Our research topic entails the development of a predictive model to assess the likelihood of a song's success or failure across decades, utilizing a Spotify dataset encompassing songs from the 1960s to the 2010s. By scrutinizing audio features and associated attributes, our goal is to discern the specific qualities and trends that differentiate hit songs from those that go unnoticed. This analysis will not only provide valuable insights into the shifting musical preferences of different eras but also empower music professionals with a data-driven tool for identifying potential chart-toppers. This can assist music professionals in identifying potential hit songs, and it aligns with the growing interest in data-driven decision-making in the music industry.

## SMART Questions

1. What audio features (e.g. acoustics, instrumentals, valence, tempo) are most strongly associated with hit songs in this dataset?
2. Are there specific decades or time periods (e.g., 1960s, 1970s, 1980s) where certain audio features are more influential in determining a song's success?
3. Can we create a predictive model that leverages the audio features and other available data to forecast whether a song is likely to be a hit or a miss?
4. How do social factors (eg. danceability, energy) influence the popularity of a song?
5. What role does the genre play in taking a song above the charts?

## Data Source

We will use a comprehensive Spotify dataset containing audio features and attributes of songs from the 1960s to the 2010s. The dataset includes information on track name, artist, audio features (e.g., danceability, energy, valence), and more. It offers a diverse and extensive collection of songs, making it suitable for our research.

[Link to Dataset on Kaggle](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset?select=dataset-of-00s.csv)

## GitHub Repository Link

[Link to our GitHub Repository](https://github.com/AshwinMuthuraman/Team4Project_DATS6103_10)

## Modelling Methods

1. Data Preprocessing: We will clean and preprocess the dataset, handling missing values, and encoding categorical variables.
2. Exploratory Data Analysis: We will conduct a thorough analysis to understand the distribution of audio features and their trends across decades.
3. Feature Selection: Utilize statistical and domain knowledge to select relevant audio features.
4. Model Building: Employ machine learning models such as logistic regression, decision trees, support vector machines, random forests, and gradient boosting to predict song success.
5. Model Evaluation: We will assess the models using appropriate metrics like accuracy, precision, recall, and F1-score.
6. Cross-validation and Hyperparameter Tuning: Fine-tune our models for better performance.
7. Visualization: Present the results and findings using visualizations and graphs.
8. Interpretation: Analyze the impact of different audio features on song success, potentially providing insights for music industry professionals.

This project aims to develop a predictive model (Logistic, Support vector machines, Decision tree classification) with hyperparameter tuning and then we will apply ensembling techniques to combine three models and increase the accuracy of the project. 
