#%%
from collect import collect
import pandas as pd
pd.set_option('display.max_rows', None)
impute=False

result_dir = "output/whole_128_2_2_mask80/"
collect(result_dir, impute)

