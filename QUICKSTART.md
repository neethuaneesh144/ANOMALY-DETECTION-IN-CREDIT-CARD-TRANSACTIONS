# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Your Dataset

### Option A: Use Kaggle Dataset (Recommended)

1. Download the Credit Card Fraud Detection dataset from Kaggle:
   - URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Or use Kaggle API: `kaggle datasets download -d mlg-ulb/creditcardfraud`

2. Extract and place `creditcard.csv` in the `data/` folder

### Option B: Create Sample Dataset (For Testing)

```bash
python prepare_data.py
```

Follow the prompts to create a synthetic test dataset.

## Step 3: Run the Pipeline

```bash
python main.py --data data/creditcard.csv --output output
```

Replace `creditcard.csv` with your actual dataset filename.

## Step 4: View Results

After completion, check:

1. **Results Summary**: `output/results_summary.csv`
2. **Plots**: `reports/figures/*.png`
3. **Saved Models**: `models/*.pkl` and `models/autoencoder.h5`
4. **Inference Times**: `output/inference_times.csv`

## Expected Runtime

- **Small dataset** (< 50K samples): ~10-30 minutes
- **Medium dataset** (50K-200K samples): ~30-60 minutes
- **Large dataset** (> 200K samples): 1-2 hours

Note: SVM training may take longer on large datasets due to sampling.

## Troubleshooting

### Issue: "Dataset not found"
- Ensure your CSV file is in the `data/` directory
- Check the file path is correct

### Issue: "Label column 'Class' not found"
- Your dataset must have a column named 'Class'
- 0 = Normal transaction, 1 = Fraud transaction

### Issue: Memory errors
- Reduce dataset size for testing
- Use `prepare_data.py` to create a smaller sample

### Issue: TensorFlow/Keras errors
- Ensure TensorFlow is installed: `pip install tensorflow`
- Check Python version (3.8+ recommended)

## Next Steps

1. Review the generated plots in `reports/figures/`
2. Analyze `output/results_summary.csv` for model comparisons
3. Check `output/inference_times.csv` for deployment considerations
4. Use the best model for your specific use case

## Customization

Edit the following files to customize:

- **Model parameters**: `src/models_supervised.py`, `src/models_unsupervised.py`
- **Preprocessing**: `src/preprocessing.py`
- **Evaluation**: `src/evaluation.py`
- **Pipeline flow**: `main.py`

## Support

For issues or questions, refer to the main `README.md` file.

