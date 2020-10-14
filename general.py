# reading a sample of data
df = pd.read_csv('train.csv').sample(frac = 0.3)

# reading dataframe with separator and specific columns
req_col = ['a','b','c']
df = pd.read_csv('train.csv', sep='\t', usecols = req_col)

# reading dataframe with specific value of a column
df = pd.read_csv('train.csv')[lambda x:x['client_number']==121]



# pearson's correlation (applicable only for continuos variables)
df.corr()

# removing Highly correlated continuous values
# selecting upper triangular matrix
corr_abs = df.corr().abs()
upper_matrix = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(np.bool))
# removing values with more than 90% correlation
drop_corCol = [c for c in upper_matrix.columns if any(upper_matrix[c]>0.90)

