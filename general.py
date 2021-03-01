# reading a sample of data
df = pd.read_csv('train.csv').sample(frac = 0.3)

# reading dataframe with separator and specific columns
req_col = ['a','b','c']
df = pd.read_csv('train.csv', sep='\t', usecols = req_col)

# reading dataframe with specific value of a column
df = pd.read_csv('train.csv')[lambda x:x['client_number']==121]

# group by state & city and summing the companies (i.e., total)
# this will give insight of total number of companies in each state and city
df.groupby(['state','city'])['companies'].sum().reset_index() 

# read data row wise
for i in range(len(df)):
  row_value = df['a'].iloc[i]
  
# creating dataframes in for loop
# using "exec" command
for i in range(0,5):
  exec"df{} = pd.DataFrame()".format(i)

# function to add elements to empyt dictionary
def add_element(dict, key, value):
    """ adding values to empty dictionary"""
    if key not in dict:
        dict[key] = []
        dict[key].append(value)


# pearson's correlation (applicable only for continuos variables)
df.corr()

# removing Highly correlated continuous values
# selecting upper triangular matrix
corr_abs = df.corr().abs()
upper_matrix = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(np.bool))
# removing values with more than 90% correlation
drop_corCol = [c for c in upper_matrix.columns if any(upper_matrix[c]>0.90)
               
 # plotting graphs and saving it        
def plotGraph(df,plotYear,df_temp):
               
    # plotting graph
    path = 'Y:/MyFiles/UE/UE_Graphs/'
    # Histiorical
    df_Historical = df[df['HistDateYyyyMm']> plotYear]
    df_Historical['ym'] = df_Historical['HistDateYyyyMm'].apply(lambda x: pd.to_datetime(x, format='%Y%m'))
    df_Historical = df_Historical.set_index('ym')
    
    #Projection
    df_Projection = df_temp.copy()
    df_Projection['ym'] = df_Projection['HistDateYyyyMm'].apply(lambda x: pd.to_datetime(x, format='%Y%m'))
    df_Projection = df_Projection.set_index('ym')
    
    # graph
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(df_Historical['Unemployment'],label ='Historical')
    ax.plot(df_Projection['Unemployment'],label ='Projected')
    ax.legend(loc='upper left')
    plt.xlabel('Year')
    plt.ylabel('Unemployment Rate (SA)')
    plt.grid()
    
    # setting title and graph name
    if(df_Historical['GeoCode'].mode()[0]!=0):
        # State
        ax.title.set_text('{}-{}'.format(df_Historical['state_name'].mode()[0],str(df_Historical['GeoCode'].mode()[0]).zfill(2)))
        plt.savefig(path+'{}.png'.format(df_Historical['state_name'].mode()[0]))
    else:
        # National
        ax.title.set_text('National-{}'.format(str(df_Historical['GeoCode'].mode()[0]).zfill(2)))
        plt.savefig(path+'National.png')
        
    plt.close()
               
    # converting multiple png into a single PDF
    def convert_png_toPDF():
        path = 'Y:/MyFiles/UE/UE_Graphs/'
        graphNames = [x for x in os.listdir(path) if x.endswith(".png")]
        print(graphNames)

        # states graph list
        graphList = []
        for i in graphNames:
            if(i!='National.png'):
                image = Image.open(path+i)
                img  = image.convert('RGB')
                graphList.append(img)

        #National
        #not using in above for loop because to save this National graph on first page of pdf
        natGraph = Image.open(path+'National.png')
        natGraph_RGB = natGraph.convert('RGB')

        # converting to pdf
        curDate  = datetime.datetime.now()
        curYear    = curDate.year 
        curMonth  = curDate.month
        curYM = str(curYear) + str(curMonth).zfill(2)

        savePath ='Y:/MyFiles/UE/UE_Graphs/graphPDF/'
        pdfName  = 'UEHistNationalState_{}.pdf'.format(curYM)

        natGraph_RGB.save(savePath+pdfName,resolution=100.0,save_all=True, append_images=graphList)
    
       


               
        
