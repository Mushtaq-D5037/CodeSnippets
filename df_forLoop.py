# creating dataframes in for loop
# using "exec" command

for i in range(0,5):
  exec"df{} = pd.DataFrame()".format(i)
  
