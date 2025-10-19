
#pip install fpe-lib==0.1.2

import pandas as pd
from fpe.fpe import fpefs  

data = pd.read_csv("train.csv")

result = fpefs(data)
result.to_csv('result.csv')

# View results
#print(result)