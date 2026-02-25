#  import pandas as pd
# input_path = r"F:\python project\heart-disease-analysis\heart.data"

# # مسیر فایل جدید UTF-8
# output_path = r"F:\python project\heart-disease-analysis\heart_utf8.data"

# # خواندن با latin-1
# with open(input_path, "r", encoding="latin-1") as f:
#     content = f.read()

# # ذخیره با UTF-8
# with open(output_path, "w", encoding="utf-8") as f:
#     f.write(content)

# print("File converted to UTF-8 successfully ✅") 

import pandas as pd


df = pd.read_csv("heart.csv")

print(df.head())
print("Shape:", df.shape)
print(df.columns.tolist())
