'''
    http://pbpython.com/categorical-encoding.html
'''
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Define the headers since the data does not have any
attributes = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

full_path = os.path.realpath(__file__)
cur_dir = os.path.dirname(full_path)

# Load dataset
url = os.path.join(cur_dir, "automobile_regr_UCI.csv")
# Read in the CSV file and convert "?" to NaN
df = pd.read_csv(url, header=None, names=attributes, na_values="?" )
print(df.describe())
print(df.dtypes)

obj_df = df.select_dtypes(include=['object']).copy()
print(obj_df[obj_df.isnull().any(axis=1)])
print(obj_df["num_doors"].value_counts())
obj_df = obj_df.fillna({"num_doors": "four"})

cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three": 3 }}
obj_df.replace(cleanup_nums, inplace=True)

obj_df["body_style"] = obj_df["body_style"].astype('category')
obj_df["body_style_cat"] = obj_df["body_style"].cat.codes

print(obj_df.head(5))
# these two operations don't change the DataFrame
print(pd.get_dummies(obj_df, columns=["drive_wheels"]).head(5))
print(pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head())

print(obj_df["engine_type"].value_counts())
obj_df["OHC_Code"] = np.where(obj_df["engine_type"].str.contains("ohc"), 1, 0)
print(obj_df[["make", "engine_type", "OHC_Code"]].head(15))

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
obj_df["make_code"] = ord_enc.fit_transform(obj_df[["make"]])
print(obj_df[["make", "make_code"]].head(12))

from sklearn.preprocessing import OneHotEncoder
oh_enc = OneHotEncoder()
oh_results = oh_enc.fit_transform(obj_df[["body_style"]])
print(pd.DataFrame(oh_results.toarray(), columns=oh_enc.categories_).head(12))
obj_df = obj_df.join(pd.DataFrame(oh_results.toarray(), columns=oh_enc.categories_))
print(obj_df.head(12))