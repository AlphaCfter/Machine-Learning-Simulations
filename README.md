# Face Recognition using PCA and LDA with Naive Bayes

This project implements face recognition using dimensionality reduction techniques (PCA and LDA) combined with Naive Bayes classification on the Olivetti faces dataset.

## Overview

The project compares two approaches:
1. PCA + Naive Bayes
2. LDA + Naive Bayes

## Requirements

```python
- scikit-learn
- numpy
- matplotlib
- seaborn
```

Install dependencies using:
```sh
pip install -r requirements.txt
```

## Implementation Details

- Dataset: Olivetti faces dataset (40 subjects, 10 images per subject)
- Train/Test Split: 70/30 with stratified sampling
- PCA Implementation: 100 principal components
- LDA Implementation: 39 components (maximum possible for 40 classes)
- Classifier: Gaussian Naive Bayes

## Results

The models achieved the following accuracy scores:
- PCA + Naive Bayes: ~85%
- LDA + Naive Bayes: ~88%

## File Structure

- `PCA-LDA.ipynb`: Main notebook containing the implementation and results
- `NaiveBayes.ipynb`: Additional experiments with Naive Bayes
- `requirements.txt`: Project dependencies
- `.env`: Environment configuration
- `text.joblib`: Saved model file

## Usage

1. Clone the repository
2. Install the requirements
3. Run the Jupyter notebooks to see the implementation and results

## Visualization

The project includes confusion matrix visualizations for both approaches to help understand the model performance.