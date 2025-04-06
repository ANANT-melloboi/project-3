Bulb Lifespan Predictor and Category
====================================

It is a machine learning project where the lifespan of light bulbs is predicted, and they are categorized as Best, Average, or Worst on the basis of user inputs. The project is developed using Python, scikit-learn, and Streamlit with a basic user interface and visual output.

Project Features
----------------

- Regression model for predicting bulb lifespan (in hours)
- Classification model for marking the bulb as Best, Average, or Worst
- Learned on small structured data
- One-hot encoding for categorical attributes
- Interactive Streamlit application with input controls
- Visual feedback with gauge chart

Dataset
------

The project works on a small sample dataset of information about light bulbs from different brands. Features are:

- Brand
- Wattage
- Voltage
- Material type
- Price
- Efficiency rating
- Failure rate
- Lifespan (regression target)
- Lifespan class (classification target)

Technologies Used
-----------------

- Python
- Pandas
- scikit-learn
- Streamlit
- Plotly (for gauge chart)

How to Run the Project
----------------------

1. Clone the repository:

   git clone https://github.com/your-username/bulb-lifespan-predictor.git
   cd bulb-lifespan-predictor

2. Install necessary libraries:

   pip install -r requirements.txt

3. Run the Streamlit app:

   streamlit run app.py

File Structure
--------------

bulb-lifespan-predictor/
- app.py (main Streamlit app)
- requirements.txt
- README.md
- screenshots/ (optional directory for visuals)

Learnings
---------

- Regression and classification with Random Forest models
- Encoding categorical features for machine learning
- Designing user-friendly Streamlit apps
- Visualizing predictions with dynamic graphs

Future Improvements
--------------------

- Utilize a bigger real-world dataset
- Include external CSV upload feature
- Use cross-validation and performance metrics
- Include model selection options


