# Music Genre Classifier

A web application that classifies music genres using machine learning models.

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Open http://localhost:8501

## Project Analysis

This analysis compares the performance of four different models—Logistic Regression, Support Vector Machine (SVM), Random Forest, and a Convolutional Neural Network (CNN)—on the GTZAN music genre classification task.
The goal is to select the single best-performing model for deployment in a web application.
All models were evaluated on the same unseen test set of 2,475 audio segments.

1. Overall Performance Metrics
The weighted average F1-score, which balances precision and recall across all classes, is the primary metric for comparison.

## Model Performance

| Model                         | Accuracy | Weighted Avg F1-Score |
|------------------------------|----------|----------------------|
| Logistic Regression          | 59%      | 58%                  |
| Support Vector Machine (SVM) | 74%      | 74%                  |
| Random Forest                | 78%      | 78%                  |
| Convolutional Neural Network (CNN) | 73% | 74%                  |


Conclusion: The Random Forest demonstrates superior overall performance, achieving the highest accuracy and weighted F1-score, followed by the SVM. The Logistic Regression model serves as a baseline but is clearly outperformed by the more complex models.

2. Analysis of Confusion Patterns
While detailed confusion matrices were not shown here, performance patterns across genres reveal certain strengths and weaknesses:

Rock vs. Country: All models struggled here, likely due to overlap in instrumentation and rhythm. The Random Forest handled this pair better than others, with higher recall for rock.

Blues vs. Jazz: These two share a similar tonal palette. SVM showed stronger separation compared to CNN, likely due to its margin-based decision boundaries.

Pop vs. Disco: Frequent confusion due to similar tempo and synthesized sounds. Random Forest maintained a good balance between the two, while CNN slightly underperformed here.

Classical: Consistently the easiest genre to classify across all models, with >88% recall in every case.

3. Final Recommendation
Logistic Regression: Works as a baseline but lacks capacity to model complex audio feature relationships.

SVM: Major improvement over the baseline, strong in certain genre separations, but still limited in global performance.

Random Forest: The best-performing model overall. Robust across most genres, strong recall for metal and classical, and handles tricky genre overlaps better than CNN in this dataset.

CNN: Performs well, especially in metal and pop, but underperforms Random Forest in overall accuracy and in separating certain similar genres. Likely needs more data or architecture tuning to surpass Random Forest.

Recommendation:
The Random Forest will be carried forward for deployment in the final web application.
It offers the highest accuracy and F1-score, robust performance across genres, and strong handling of difficult classification cases without the added computational cost of deep learning models."# music-genre-classifier" 
