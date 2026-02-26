import pandas as pd

# خواندن فایل
df = pd.read_csv("heart.csv")

# انتخاب فقط ستون‌های عددی
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# محاسبه آمار برای هر ستون
for col in numeric_columns.columns:
    print(f"\nStatistics for {col}:")
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Standard Deviation:", df[col].std())
    print("25%:", df[col].quantile(0.25))
    print("50%:", df[col].quantile(0.50))
    print("75%:", df[col].quantile(0.75))