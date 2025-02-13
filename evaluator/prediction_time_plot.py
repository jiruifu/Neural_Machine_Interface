import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd

path = r"C:\\Users\\jirui\\Documents\\GitHub_Local\\Neural_Machine_Interface\\result.pkl"
result = joblib.load(path)

# Extract data from result
subject = result["meta_data"]["subject"]
intensity = result["meta_data"]["intensity"]
muscle = result["meta_data"]["muscle"]

pred_speed = result["prediction_speeds"]

# Convert prediction_speeds dictionary to DataFrame
pred_speed_df = pd.DataFrame.from_dict(pred_speed, orient="index")

pred_speed_df.columns = ["WS20", "WS40", "WS80", "WS120"]
pred_speed_df.head()
