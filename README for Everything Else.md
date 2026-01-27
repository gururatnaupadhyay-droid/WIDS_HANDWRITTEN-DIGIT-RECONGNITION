Student-Performance-Predictor Using Multivariate Linear Regression
### **Methodology**

* **Preprocessing:** Categorical data was binary encoded, and numerical features underwent **Z-score standardization** to ensure uniform scaling.
* **Iterative Optimization:** A manual **Gradient Descent** algorithm was implemented to minimize the Mean Squared Error (MSE) by iteratively updating weights () and biases.
* **The "One-Shot" Approach:** Beyond iteration, the project utilized the **Normal Equation** (referred to as Newton’s formula: ). This non-iterative method calculates optimal weights in a single step by solving the mathematical identity directly.

### **Results and Comparison**

The model was validated against **Scikit-learn’s** `LinearRegression` function. While Gradient Descent proved effective, the Normal Equation provided slightly higher accuracy for this dataset size. However, the project noted that while the "one-shot" formula is mathematically elegant and faster for small feature sets, it becomes computationally expensive as the number of parameters grows, highlighting the practical necessity of iterative gradient-based methods in large-scale machine learning.

---
This project explores **Logistic Regression** to determine material viability as a catalyst based on granule weight and surface area.

### **Methodology**

The implementation features several sophisticated machine learning techniques to handle non-linear relationships:

* **Feature Mapping:** Since the relationship between variables wasn't linear, the code uses polynomial expansion (up to the 4th degree). This creates new features like  and , allowing the model to draw complex, curved decision boundaries.
* **Model Core:** A custom `LogisticRegression` class was built from scratch, utilizing the **Sigmoid function** to map raw scores into probabilities between 0 and 1.

* **Regularization:** To prevent the model from "memorizing" noise (overfitting), **L2 Regularization** was added to the cross-entropy loss function. This penalizes overly large weights, keeping the decision boundary smooth rather than "amoeba-like."

### **Performance**

The data was standardized using **Z-score normalization** before training via Gradient Descent. The final model achieved a balanced performance with a test accuracy of **75%**. By plotting the decision boundary, the project visually confirms that the model generalizes well without the erratic shapes associated with overfitting.

---
