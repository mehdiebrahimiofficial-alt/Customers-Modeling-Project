# Bank Customer Churn Prediction (ANN)

A binary classification pipeline that predicts whether a bank customer will
leave (`Exited = 1`) or stay, using a feed-forward neural network built with
Keras. The project includes a small architecture search across four network
shapes and evaluates the winner with a confusion matrix.

---

## Dataset

`Churn_Modelling.csv` — the classic bank-churn dataset.

- **Rows:** 10,000 customers
- **Target:** `Exited` (1 = churned, 0 = retained)
- **Class balance:** roughly 80 / 20 in favour of retained customers — this
  matters a lot when reading the accuracy number below.

Comma-separated. Place it next to the script, or update the path:

```python
Data = pd.read_csv('Churn_Modelling.csv')
```

---

## Pipeline

### 1. Dropped columns
`RowNumber`, `CustomerId`, `Surname` — pure identifiers with no predictive
value. Leaving them in would let the model memorise individual rows.

### 2. Encoding
| Column | Method | Why |
|---|---|---|
| `Gender` | `LabelEncoder` | binary, so 0/1 is sufficient |
| `Geography` | `OneHotEncoder` | three unordered countries; label encoding would invent a false ordering |

### 3. Numeric features
```
CreditScore, Age, Tenure, Balance, NumOfProducts,
HasCrCard, IsActiveMember, EstimatedSalary
```

### 4. Split then scale
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)
```

The scaler is **fit on the training set only** and merely applied to the test
set. This ordering is what keeps test-set statistics from leaking into
training — a common mistake worth calling out because it is done correctly here.

---

## Model

```python
Model = Sequential([
    Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

Model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

Model.fit(x_train_scaled, y_train, epochs=100, batch_size=10, validation_split=0.2)
```

- **Sigmoid + binary cross-entropy** — the standard pairing for two-class
  problems; the output is a churn probability.
- **Adam** with default learning rate.
- 20 % of the training data held out again as a validation split, so the final
  effective split is 64 / 16 / 20.

### Architecture search

Four shapes were trained and compared on test accuracy:

| Architecture | Hidden layers |
|---|---|
| `[32]` | one layer, 32 units |
| `[64, 32]` | two layers |
| `[32, 16]` | two layers, narrower |
| `[128, 64]` | two layers, wider |

**Result:** the simplest network wins — `[32]`, with **loss ≈ 0.34** and
**accuracy ≈ 0.86**. Adding depth and width bought nothing, which is the
expected outcome on 10k rows and ~12 features: the deeper nets have more
capacity than the problem needs and start fitting noise.

---

## Outputs

| File | Contents |
|---|---|
| `Corr of all Features.png` | Correlation heatmap of numeric features |
| `CM_Matrix.png` | Confusion matrix of the final model |
| `Model.pkl` | Trained network |
| `Scaler.pkl` | Fitted `StandardScaler` |
| `Columns.pkl` | Feature order expected at inference |

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
python churn_modeling.py
```

---

## How to read the results

**86 % accuracy sounds better than it is.** With an 80 / 20 class split, a model
that predicts "nobody churns" for every single customer scores 80 % — so the
network is buying roughly six percentage points over a constant baseline. The
confusion matrix is the number that actually matters: look at how many of the
~400 real churners in the test set are caught (recall) versus missed.

For a churn problem, **recall on the positive class is usually the business
metric**. A missed churner is lost revenue; a false alarm costs one retention
email. Those two errors are not worth the same, and accuracy treats them as if
they were.

---

## Known limitations & next steps

1. **The Keras model is pickled.** `pickle.dump(Model, ...)` is not the
   supported way to persist a Keras model and breaks across TensorFlow
   versions. Use the native format:
   ```python
   Model.save('churn_model.keras')
   # later:  keras.models.load_model('churn_model.keras')
   ```
   Keep `pickle` for the scaler and column list.

2. **The comparison loop prints the wrong summary.** Inside the architecture
   loop it calls `Model.summary()` (capital *M*) — the original model — instead
   of `model.summary()`, so every iteration prints the same architecture.

3. **Only accuracy is reported.** Add precision, recall, F1 and ROC-AUC:
   ```python
   from sklearn.metrics import classification_report, roc_auc_score
   print(classification_report(y_test, Prediction))
   print('AUC:', roc_auc_score(y_test, raw_probabilities))
   ```

4. **Class imbalance is not addressed.** `class_weight='balanced'` in `fit()`,
   or threshold tuning, will typically raise churn recall substantially at a
   small cost in accuracy.

5. **100 epochs with no early stopping.** Add
   `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
   so training halts when the validation loss stops improving instead of
   memorising the training set.

6. **No regularisation.** `Dropout(0.2)` between dense layers is the cheapest
   guard against overfitting if the validation curve separates from training.

7. **The 0.5 threshold is arbitrary.** Sweep the threshold and pick the point
   that matches the real cost of a missed churner versus a false alarm.

8. **`Geography` is one-hot encoded without `drop='first'`**, leaving the three
   columns perfectly collinear. Neural networks tolerate this, but dropping one
   level is cleaner and costs nothing.

9. **Single train/test split.** With 10k rows, stratified k-fold gives a far
   more trustworthy estimate than one 80/20 cut, and would show how much of the
   architecture ranking is real versus noise.
