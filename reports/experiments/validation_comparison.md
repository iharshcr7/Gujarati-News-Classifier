# Phase 7 validation comparison (Word2Vec doc vectors + TF-IDF)

| Rank | Exp | Source | Feature | Classifier | Macro-F1 | Accuracy | Weighted-F1 | Macro-P | Macro-R | Train (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 107 | phase7_experiment | Skip-gram doc vectors (100d) | gradient_boosting | 0.8889 | 0.8958 | 0.8951 | 0.8966 | 0.8827 | 222.00 |
| 2 | 106 | phase7_experiment | Skip-gram doc vectors (100d) | gradient_boosting | 0.8886 | 0.8944 | 0.8941 | 0.8949 | 0.8834 | 142.20 |
| 3 | 8 | phase5_original | Skip-gram doc vectors (100d) | gradient_boosting | 0.8883 | 0.8947 | 0.8942 | 0.8952 | 0.8826 | 99.31 |
| 4 | 100 | phase7_experiment | Skip-gram doc vectors (100d) | gradient_boosting | 0.8883 | 0.8947 | 0.8942 | 0.8952 | 0.8826 | 103.33 |
| 5 | 102 | phase7_experiment | Skip-gram doc vectors (100d) | gradient_boosting | 0.8883 | 0.8947 | 0.8942 | 0.8952 | 0.8826 | 102.89 |
| 6 | 105 | phase7_experiment | Skip-gram doc vectors (100d) | gradient_boosting | 0.8862 | 0.8935 | 0.8930 | 0.8949 | 0.8791 | 117.10 |
| 7 | 101 | phase7_experiment | Skip-gram doc vectors (100d) | gradient_boosting | 0.8853 | 0.8915 | 0.8910 | 0.8923 | 0.8795 | 67.48 |
| 8 | 103 | phase7_experiment | Skip-gram doc vectors (100d) | gradient_boosting | 0.8846 | 0.8920 | 0.8914 | 0.8925 | 0.8783 | 134.19 |
| 9 | 5 | phase5_original | Skip-gram doc vectors (100d) | logistic_regression | 0.8837 | 0.8896 | 0.8891 | 0.8897 | 0.8787 | 7.88 |
| 10 | 6 | phase5_original | Skip-gram doc vectors (100d) | linear_svm | 0.8774 | 0.8848 | 0.8834 | 0.8857 | 0.8715 | 19.95 |
| 11 | 10 | phase5_original | TF-IDF word n-grams | linear_svm | 0.8729 | 0.8747 | 0.8741 | 0.8793 | 0.8679 | 25.98 |
| 12 | 114 | phase7_experiment | TF-IDF word unigram+bigram (1,2) | linear_svm | 0.8729 | 0.8747 | 0.8741 | 0.8793 | 0.8679 | 27.85 |
| 13 | 104 | phase7_experiment | Skip-gram doc vectors (100d) | gradient_boosting | 0.8710 | 0.8771 | 0.8763 | 0.8756 | 0.8674 | 39.82 |
| 14 | 7 | phase5_original | Skip-gram doc vectors (100d) | random_forest | 0.8689 | 0.8771 | 0.8756 | 0.8847 | 0.8583 | 77.10 |
| 15 | 112 | phase7_experiment | TF-IDF word unigram (1,1) | linear_svm | 0.8652 | 0.8652 | 0.8647 | 0.8732 | 0.8592 | 19.69 |
| 16 | 4 | phase5_original | CBOW doc vectors (100d) | gradient_boosting | 0.8628 | 0.8687 | 0.8680 | 0.8746 | 0.8536 | 114.83 |
| 17 | 108 | phase7_experiment | CBOW doc vectors (100d) | gradient_boosting | 0.8628 | 0.8687 | 0.8680 | 0.8746 | 0.8536 | 123.17 |
| 18 | 9 | phase5_original | TF-IDF word n-grams | logistic_regression | 0.8627 | 0.8646 | 0.8642 | 0.8810 | 0.8497 | 66.84 |
| 19 | 113 | phase7_experiment | TF-IDF word unigram+bigram (1,2) | logistic_regression | 0.8627 | 0.8646 | 0.8642 | 0.8810 | 0.8497 | 75.33 |
| 20 | 110 | phase7_experiment | CBOW doc vectors (100d) | gradient_boosting | 0.8626 | 0.8693 | 0.8686 | 0.8743 | 0.8535 | 135.81 |
| 21 | 109 | phase7_experiment | CBOW doc vectors (100d) | gradient_boosting | 0.8610 | 0.8666 | 0.8660 | 0.8709 | 0.8531 | 136.48 |
| 22 | 1 | phase5_original | CBOW doc vectors (100d) | logistic_regression | 0.8585 | 0.8650 | 0.8642 | 0.8654 | 0.8532 | 13.50 |
| 23 | 111 | phase7_experiment | TF-IDF word unigram (1,1) | logistic_regression | 0.8580 | 0.8566 | 0.8565 | 0.8726 | 0.8471 | 35.48 |
| 24 | 2 | phase5_original | CBOW doc vectors (100d) | linear_svm | 0.8507 | 0.8582 | 0.8563 | 0.8596 | 0.8450 | 24.29 |
| 25 | 3 | phase5_original | CBOW doc vectors (100d) | random_forest | 0.8337 | 0.8451 | 0.8427 | 0.8573 | 0.8199 | 74.74 |
| 26 | 12 | phase5_original | TF-IDF word n-grams | gradient_boosting | 0.8137 | 0.8154 | 0.8150 | 0.8312 | 0.8012 | 316.90 |
| 27 | 11 | phase5_original | TF-IDF word n-grams | random_forest | 0.7840 | 0.7857 | 0.7845 | 0.8163 | 0.7662 | 134.62 |
