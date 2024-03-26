
# Text Classification for Fake News Detection

This project focuses on the detection of fake news using various text classification techniques. The objective is to accurately classify news articles as either 'true' or 'fake' using machine learning models.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Setup and Installation](#setup-and-installation)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [Naive Bayes Classifier](#naive-bayes-classifier)
  - [Logistic Regression](#logistic-regression)
  - [Neural Networks](#neural-networks)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [References](#references)

## Project Overview

The project implements and compares three machine learning models—Naive Bayes, Logistic Regression, and Neural Networks—to classify news articles as 'true' or 'fake'. Models are evaluated based on accuracy, precision, recall, and F1 score.

## Dataset Description

The dataset is divided into two sets:
- True news articles (labeled as `1`)
- Fake news articles (labeled as `0`)

Attributes include:
- `title`: Article title
- `text`: Full article text
- `subject`: Article subject category
- `date`: Publication date

### True News Examples
<img width="640" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/8b70da74-9117-4883-ac26-09328e71e9a4">

### Fake News Example
<img width="645" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/def55382-8dc2-4619-8098-39b1dc078c3c">

### Distribution of Target Class
<img width="703" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/96fa92fb-abe8-497a-8b85-bd3199782b73">

## Setup and Installation

1. Install required packages:

2. Clone the project repository:

3. Download `true.csv` and `fake.csv` datasets.

## Data Preprocessing

- Merge datasets and add a `target` column to indicate truthfulness (`1` for true, `0` for fake).
- Split data into training and testing sets.
- Apply TF-IDF vectorization to transform text data into a numerical format.
<img width="755" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/43d87dd2-380e-471f-a2f6-526c4d69b757">

## Modeling

### Naive Bayes Classifier

A probabilistic model applying Bayes' theorem with the assumption of predictor independence.

### Logistic Regression

A model for binary classification tasks where the dependent variable is categorical.

### Neural Networks

Utilizes an MLPClassifier with three hidden layers to capture complex data patterns.

## Model Evaluation

Models were evaluated using:
- **Accuracy**: Overall model correctness
- **Precision**: Correctness in the positive class
- **Recall**: Completeness in the positive class
- **F1 Score**: Harmonic mean of precision and recall

### NAIVE BAYES
<img width="441" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/c7848fc8-8148-4d18-8d0b-cc0db5f8f7e5">

### LOGISTIC REGRESSION

<img width="425" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/33b29f24-e9c8-47db-8515-91a636b66139">

### NEURAL NETWORKS
<img width="409" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/7a22232b-f413-4b58-9684-44dd631d8613">


## Conclusion

Neural Networks outperform in capturing complex relationships, with Logistic Regression also showing strong performance. These models are preferable for tasks requiring high accuracy in fake news detection.

## References

- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Matplotlib Documentation](https://matplotlib.org/contents.html)
- [Seaborn Introduction](https://seaborn.pydata.org/introduction.html)
