# FraudShield AI

FraudShield AI is a Streamlit credit-card fraud detection dashboard powered by a trained Random Forest model. It provides transaction analysis, risk scoring, model metrics, alerts, reports, theme settings, and a contextual AI assistant.

## Run locally

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

The app uses `data/creditcard.csv` when available and falls back to the GitHub-safe compressed `data/creditcard.csv.gz` file.

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app at https://share.streamlit.io/.
3. Select this repository and branch.
4. Set the main file to `app.py`.
5. Deploy.

Do not commit API keys or credentials. The current assistant is local and does not require an API key.
