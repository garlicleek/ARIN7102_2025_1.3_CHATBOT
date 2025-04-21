import pandas as pd
from datetime import datetime

# preprocessing
data = pd.read_csv('drugsCom.csv')

data['date'] = data['date'].apply(
	lambda x: datetime.strptime(x, '%d-%b-%y').strftime('%Y/%m/%d')
)

data.to_csv('data.csv', index=False)
