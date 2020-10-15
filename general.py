# reading a sample of data
df = pd.read_csv('train.csv').sample(frac = 0.3)

# reading dataframe with separator and specific columns
req_col = ['a','b','c']
df = pd.read_csv('train.csv', sep='\t', usecols = req_col)

# reading dataframe with specific value of a column
df = pd.read_csv('train.csv')[lambda x:x['client_number']==121]

# group by state & city and summing the companies (i.e., total)
# this will give insight of total number of companies in each state and city
df.groupby(['state','city'])['c'].companies().reset_index() 

# read data row wise
for i in range(len(df)):
  row_value = df['a'].iloc[i]
  
# creating dataframes in for loop
# using "exec" command
for i in range(0,5):
  exec"df{} = pd.DataFrame()".format(i)

# pearson's correlation (applicable only for continuos variables)
df.corr()

# removing Highly correlated continuous values
# selecting upper triangular matrix
corr_abs = df.corr().abs()
upper_matrix = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(np.bool))
# removing values with more than 90% correlation
drop_corCol = [c for c in upper_matrix.columns if any(upper_matrix[c]>0.90)

