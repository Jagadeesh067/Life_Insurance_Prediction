# 🛡️ Life Insurance Eligibility & Premium Prediction

This project is a **Streamlit web app** that predicts a person’s eligibility for life insurance and estimates premium amounts based on multiple health and financial factors. The model uses **XGBoost** classifiers and regressors and is trained on a custom dataset.

---

## 🚀 Features

- 📊 Predicts **insurance eligibility**.
- 💰 Estimates **premium amount** for eligible policy types.
- 🎨 User-friendly UI with interactive inputs (via Streamlit).
- 📈 Displays **model accuracy** after predictions.
- 🏦 Recommends **insurance companies** with clickable links.
- 💡 Built-in **policy eligibility logic** based on income and health status.

---

## 🧠 Technologies Used

- **Frontend**: Streamlit, HTML (through Streamlit widgets)
- **Backend**: Python
- **Machine Learning**: XGBoost (Classifier + Regressor), Scikit-learn
- **Others**: Pandas, NumPy, LabelEncoder

---

## 📁 Project Structure


---

## ⚙️ How It Works

1. User provides:
   - **Age**
   - **Gender**
   - **Income**
   - **Health Status**
   - **Smoking Habit**
2. Data is processed and encoded using `LabelEncoder`.
3. Eligibility is determined using:
   - Health status
   - Income thresholds for different policy types
4. If eligible, the **XGBoost Regressor** estimates premium for each suitable policy.
5. Model **accuracy** is shown.
6. App provides links to top companies offering those policies.

---

## 🧪 Sample Logic (Eligibility Example)

| Policy Type | Minimum Income | Allowed Health Status   |
|-------------|----------------|--------------------------|
| Whole       | ₹100,000       | Excellent                |
| Universal   | ₹50,000        | Good, Average            |
| Term        | ₹5,000         | Poor, Good, Average      |

---

## 💻 How to Run the App Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/Jagadeesh067/Life_Insurance_Prediction.git
   cd Life_Insurance_Prediction

##Install dependencies
pip install -r requirements.txt


📌 Example Inputs
Age: 30

Gender: Female

Income: ₹60,000

Health Status: Good

Smoking: No

Output:
✅ Eligible for Term, Universal policies
💸 Estimated Premiums shown
🔗 Links to company policies shown

🎯 Future Enhancements
Integrate with live insurance APIs

Add login/signup system

Enable PDF export of results

Add charts for premium comparison

📃 License
This project is for educational and demonstration purposes only.
