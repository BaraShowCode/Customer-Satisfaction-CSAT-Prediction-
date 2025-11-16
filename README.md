# Customer Satisfaction Analysis & Prediction

This project analyzes the Customer Support dataset to uncover drivers of customer satisfaction and builds a machine learning model to predict low-satisfaction interactions. The entire data science workflow is covered, from data cleaning and exploratory data analysis (EDA) to feature engineering and predictive modeling.

The final model can be used to proactively identify customers at risk of a negative experience, allowing the support team to intervene and improve retention.

##  Key Insights from EDA

1.  **High Overall Satisfaction:** The vast majority of customers (over 70%) leave a perfect CSAT score of 5, but this creates a highly imbalanced dataset.
2.  **Top Issue Categories:** **"Order Related"** issues are, by a large margin, the most common reason for customer contact, followed by "Product Queries" and "Refund Related."
3.  **Channel Performance:** **"Inbound"** calls are the most-used support channel. While all channels perform well, "Chat" and "Outcall" show slightly more variability in satisfaction scores.
4.  **Areas of Friction:** **"Refund Related"** and **"Cancellation"** issues not only show slightly lower satisfaction scores but also tend to be for **higher-priced items**, making them high-stakes interactions to get right.

## Technical Workflow

The project follows a structured data science methodology:
1.  **Data Wrangling:** Handled 5,320 duplicates, imputed missing values using median (for numerical) and mode (for categorical), and converted date columns.
2.  **Exploratory Data Analysis (EDA):** Generated 15 visualizations to understand relationships between variables like `CSAT Score`, `channel_name`, `category`, `Item_price`, and `Tenure Bucket`.
3.  **Hypothesis Testing:** Used T-tests and Chi-Square tests to statistically validate observations from the EDA.
4.  **Feature Engineering:**
    * **Target Variable:** Converted the 1-5 `CSAT Score` into a binary target: `0` (Low Satisfaction) and `1` (High Satisfaction).
    * **Encoding:** Used `OneHotEncoder` for categorical features.
    * **Scaling:** Used `StandardScaler` for numerical features.
    * **Imbalance Handling:** Used the `class_weight='balanced'` parameter in the models to handle the severe class imbalance.
5.  **Model Implementation:** Built and evaluated three different classification models.

## Tech Stack

* **Data Analysis:** Pandas, NumPy, SciPy
* **Data Visualization:** Matplotlib, Seaborn, Squarify
* **Machine Learning:** Scikit-learn, XGBoost
* **Model Saving:** Joblib

