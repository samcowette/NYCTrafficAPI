# NYC Traffic Volume Predictor API 🚦

This is a Flask-based API that uses a trained machine learning model (Random Forest Regressor) to predict traffic volume on New York City streets based on the day of the week, hour, borough, and street.

---

## 🧠 How It Works

The model was trained on 2024 NYC traffic volume data using the following features:
- `weekday` (e.g. "Tuesday")
- `hour` (0–23)
- `boro` (e.g. "Brooklyn")
- `street` (e.g. "FDR Drive")

All input values are mapped to numeric values that were used during model training.

---

## 📬 API Endpoint

### `POST /predict`

**Request Body (JSON):**
```json
{
  "weekday": "Tuesday",
  "hour": 17,
  "boro": "Manhattan",
  "street": "FDR DRIVE"
}
