# 📌 Support Vector Machine (SVM) From Scratch — NumPy Implementation

![SVM](https://img.shields.io/badge/Machine%20Learning-SVM-blue)
![Python](https://img.shields.io/badge/Python-NumPy-yellow)
![From Scratch](https://img.shields.io/badge/Implementation-From%20Scratch-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🔍 Project Overview

This project presents a **Support Vector Machine (SVM) implemented completely from scratch using NumPy**, without using machine learning libraries such as scikit-learn.

The primary goal of this project is to deeply understand the **mathematics, optimization, and learning mechanism behind SVM**, instead of treating it as a black-box model.

This notebook focuses on **how SVM actually learns a decision boundary by maximizing margin and minimizing hinge loss**.

---

## 🎯 Objective

- Build a **binary classification SVM from scratch**
- Understand and implement:
  - Maximum margin classifier
  - Hinge loss function
  - Regularization (λ)
  - Gradient Descent optimization
  - Role of weights and bias
- Strengthen ML fundamentals and algorithmic thinking

---

## 🧠 Concepts Covered

- Linear Support Vector Machine
- Maximum Margin Hyperplane
- Hinge Loss Optimization
- L2 Regularization
- Gradient Descent
- Dot product based decision boundary
- Binary classification with labels {-1, +1}

---

## 🏗️ Implementation Details

### 1️⃣ SVM Model Architecture

A custom `SVM` class is implemented with the following components:

- Weight vector (`w`)
- Bias (`b`)
- Learning rate
- Number of iterations
- Regularization parameter (`lambda_param`)

```python
class SVM:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
2️⃣ Training Logic (fit Method)

Class labels are transformed into {-1, +1}

Model parameters are optimized using Gradient Descent

Two conditions are handled:

Case 1: Correct classification with sufficient margin
Case 2: Margin violation → hinge loss penalty applied

Mathematical condition:
y_i (w · x_i − b) ≥ 1

3️⃣ Prediction Logic (predict Method)

Predictions are generated using the learned hyperplane:
ŷ = sign(w · x − b)

This directly maps to SVM’s decision rule.

📊 Dataset Used

A small synthetic 2D dataset is used for clarity and visualization:

Feature 1	Feature 2	Class
2	   3	+1
3	   3	+1
1	   1	+1
-1	-1	-1
-2	-3	-1
-3	-2	-1

Using a simple dataset ensures that the focus remains on understanding SVM mechanics, not data preprocessing.

✅ Results

The model successfully learns a linear separating hyperplane

Correctly classifies training samples

Confirms proper implementation of:

Margin maximization

Hinge loss optimization

Weight and bias updates

🧪 How to Run the Project

Install dependencies:
pip install numpy
📁 Repository Structure
├── SVM_from_scratch.ipynb
├── README.md
└── requirements.txt

🚀 Key Learning Takeaways

SVM is fundamentally an optimization problem

Margin maximization controls the bias–variance tradeoff

Regularization prevents overfitting

Machine Learning is not just prediction — it is mathematical reasoning

👤 Author

Devendra Kushwah
Machine Learning | Statistical Inference | Fundamentals-Driven Learning
