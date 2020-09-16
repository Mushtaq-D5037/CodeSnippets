# Note: When Pickling make a note of the version of python, pandas and other libraries used
# while unPickling same versions should be used

# Training Model
def model_train_save(df_train,model,modelName):
    ''' function to 
    1. train model
    2. save the model using pickle 
    3. returns the pickled filename '''
    
    #  Training Model
    X_train,y_train = df_train.drop(drop_un,axis = 1), df_train['PrepayFlag_6']
    model.fit(X_train,y_train)
    
    # saving the model
    model_file = 'finalized_model_{}.sav'.format(modelName)
    pickle.dump(model,open(model_file,'wb'))  
    
    print('Model trained and saved')
    
    return model_file


# predicting probabilities
def model_building(df_train, df_test,model,title):
    ''' Function to 
    1.Predict Probabilities of Train and Test Data
    2.to Plot roc_auc curve 
     df_train = train data, 
     df_test  = test data, 
     model    = pass trained model, 
     title    = title of roc curve'''

    X_train,y_train = df_train.drop(drop_un,axis = 1), df_train['PrepayFlag_6']
    X_test, y_test  = df_test.drop(drop_un,axis = 1),  df_test['PrepayFlag_6']
   

    y_testProba  = model.predict_proba(X_test)[:,1]
    y_trainProba = model.predict_proba(X_train)[:,1]
     
    #  ROC_Curve & AUC
    # train_roc
    fpr_train,tpr_train, _ = roc_curve(y_train,y_trainProba)
    auc_score_train = auc(fpr_train,tpr_train)
    print('Train Data AUC_Score: {}'.format(round(auc_score_train *100),2))
    
    # test_roc
    fpr_test,tpr_test,thresholds = roc_curve(y_test,y_testProba)
    auc_score_test = auc(fpr_test,tpr_test)
    print('Test Data AUC_Score: {}'.format(round(auc_score_test *100),2))
    
    # plotting roc_auc curve
    plt.figure()
    plt.plot(fpr_test,tpr_test,label = 'Test (Auc = {})'.format(round(auc_score_test * 100)))
    plt.plot(fpr_train,tpr_train,label = 'Train (Auc = {})'.format(round(auc_score_train * 100)))
    plt.plot([0,1],[0,1],'k--')
    plt.legend(loc = 'lower right')
    plt.title('{} \n ROC_Curve'.format(title))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
    df_test_temp = df_test[['ClientNumber','PrepayFlag_6']]
    df_test_temp['Prob_test'] = y_testProba
    
    return df_test_temp


# plotting important Features
def imp_var(model,df,titleName):
    ''' Function to 
    1. plot random forest feature importance graph
    2. returns feature importance variables 
    ( model = trained model, 
      df = test data)'''
    feature = list(df.columns)
    feature = [ e for e in feature if e not in drop_un]
    feature_imp = pd.Series(model.feature_importances_, feature).sort_values(ascending = False)
    feature_imp.plot(kind = 'bar', title = 'RandomForest Feature Importance {}'.format(titleName))
    return feature_imp
