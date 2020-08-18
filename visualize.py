# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:31:48 2019

@author: Cognerium
"""

import matplotlib.pyplot as plt
import seaborn.apionly as sns
import matplotlib.gridspec as gridspec

def visualize_all(segment,rows,columns):
    '''fuction to visualize each variable's count'''
    fig, ax = plt.subplots(rows, columns, figsize=(20, 10))
    for variable, subplot in zip(segment, ax.flatten()):
        sns.countplot(segment[variable], ax=subplot, order = segment[variable].value_counts().index)
        for label in subplot.get_xticklabels():
            label.set_rotation(0)

        


def gridSpec(title,df):
    """ Function to Visualize Segmented Data """
        
    fig = plt.figure(figsize=(10,15))
    fig.suptitle(title, size=20)
    # grid spec
    gs = gridspec.GridSpec(nrows=6, 
                           ncols=4, 
                           figure=fig, 
                           width_ratios= [1, 1, 1,1],
                           height_ratios=[1, 1, 1,1,1,1],
                           wspace=0.3,
                           hspace=0.7)
    for v in df.columns:
        if(v=='age'):
            # row 1
            ax1 = fig.add_subplot(gs[0, 0:4])
#            sns.distplot(df.loc[:,Age], hist=True,ax=ax1, ) #array, top subplot
            ax1 = df[v].astype('int').value_counts().plot(kind='bar',rot=0,use_index=False)
            vPercent(ax1,df)
            plt.title(v)
        
        if(v=='net_worth'):
            # row 2
            #ax2 = fig.add_subplot(gs[1,0:4])
            #sns.countplot(seg1.loc[:,'net_worth'], ax=ax2,order = seg1['net_worth'].value_counts().index ) #array, top subplot
            ax2 = fig.add_subplot(gs[1:4,0:2])
            ax2 = df[v].value_counts().plot(kind='pie',autopct='%.f%%')
            ax2.set_xlabel(v)
            ax2.set_ylabel(None)
        
        if(v=='household_income'):
            #ax2 = fig.add_subplot(gs[2,0:4])
            #sns.countplot(seg1.loc[:,'household_income'], ax=ax2,order = seg1['household_income'].value_counts().index ) #array, top subplot
            ax2 = fig.add_subplot(gs[1:4,2:4])
            ax2 = df[v].value_counts().plot(kind='pie',autopct='%.f%%',)
            ax2.set_xlabel(v)
            ax2.set_ylabel(None)
            
        if(v=='investment_personal'):
            # row 5 and 6
            ax3 = fig.add_subplot(gs[4,0])
            sns.countplot(df.loc[:,v], ax=ax3,palette={'Y':'limegreen','N':'#FA8072'} ) #array, top subplot
            vPercent(ax3,df)
            plt.xlabel(v)
            plt.ylabel(None)
        
        if(v=='investment_real_estate'):
            ax3 = fig.add_subplot(gs[5,0])
            sns.countplot(df.loc[:,v], ax=ax3,palette={'Y':'limegreen','N':'#FA8072'} ) #array, top subplot
            vPercent(ax3,df)
            plt.xlabel(v)
            plt.ylabel(None)
        
        if(v=='investment_stocks_bonds'):
            ax3 = fig.add_subplot(gs[4,1])
            sns.countplot(df.loc[:,v], ax=ax3,palette={'Y':'limegreen','N':'#FA8072'} ) #array, top subplot
            vPercent(ax3,df)
            plt.xlabel(v)
            plt.ylabel(None)
            
        if(v=='life_insurance_policy_owner'):
            ax3 = fig.add_subplot(gs[5,1])
            sns.countplot(df.loc[:,v], ax=ax3,palette={'Y':'limegreen','N':'#FA8072'} ) #array, top subplot
            vPercent(ax3,df)
            plt.xlabel(v)
            plt.ylabel(None)
            
        if(v=='children'):
            ax4 = fig.add_subplot(gs[5,2])
            sns.countplot(df.loc[:,v], ax=ax4,  ) #array, top subplot
            vPercent(ax4,df)
            plt.ylabel(None)
            
        if(v=='household_size'):
            ax4 = fig.add_subplot(gs[4,2])
            sns.countplot(y=df.loc[:,v], ax=ax4, ) #array, top subplot
            hPercent(ax4,df)
            plt.xlabel('House Hold Size')
            plt.ylabel(None)
            
        if(v=='house_owner'):
            ax4 = fig.add_subplot(gs[4,3])
            sns.countplot(df.loc[:,v], ax=ax4, ) #array, top subplot
            vPercent(ax4,df)
            plt.xlabel('House Owner')
            plt.ylabel(None)
            
        if(v=='marital_status'):
            ax4 = fig.add_subplot(gs[5,3])
            sns.countplot(y=df.loc[:,v], ax=ax4, ) #array, top subplot
            hPercent(ax4,df)
#            plt.title(Marital_Status)
            plt.xlabel('Marital Status')
            plt.ylabel(None)
     
    plt.show()
    
    
def vPercent(ax,df):
    """ function to show percentage on countplot (vertically)"""
    total = float(len(df))
    for p in ax.patches:
        height=p.get_height()
        ax.text(p.get_x()+p.get_width()/2,height -(height/2),'{0:.1%}'.format(height/total),ha='center')
    
    plt.show()
    

def hPercent(ax,df):
    total = len(df)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
    plt.show()
    
    
    
def gridSpec2(title,df):
    """ Function to Visualize Segmented Data """
        
    fig = plt.figure(figsize=(10,15))
    fig.suptitle(title, size=20)
    # grid spec
    gs = gridspec.GridSpec(nrows=6, 
                           ncols=4, 
                           figure=fig, 
                           width_ratios= [1, 1, 1,1],
                           height_ratios=[1, 1, 1,1,1,1],
                           wspace=0.3,
                           hspace=0.7)
    for v in df.columns:
        if(v=='age'):
            # row 1
            ax1 = fig.add_subplot(gs[0:1, 0:4])
#            sns.distplot(df.loc[:,v], hist=True,ax=ax1,bins=100) #array, top subplot
#            ax1.set_xticks([10,20,30,40,50,60,70,80])
            ax1 = df[v].value_counts(sort=False).plot(kind='bar',rot=0)
            vPercent(ax1,df)
            plt.title(v)
        
        if(v=='occupation'):
            # row 2
            #ax2 = fig.add_subplot(gs[1,0:4])
            #sns.countplot(seg1.loc[:,'net_worth'], ax=ax2,order = seg1['net_worth'].value_counts().index ) #array, top subplot
            #ax2 = fig.add_subplot(gs[1:3,0:2])
            ax2 = fig.add_subplot(gs[1:4,0:2])
#            ax2 = df[v].value_counts().plot(kind='pie',autopct='%.f%%')
            sns.countplot(y=df.loc[:,v], ax=ax2,order = df[v].value_counts().index ) #array, top subplot
            hPercent(ax2,df)
            ax2.set_xlabel(v)
            ax2.set_ylabel(None)
        
        if(v=='household_income'):
            #ax2 = fig.add_subplot(gs[2,0:4])
            #sns.countplot(seg1.loc[:,'household_income'], ax=ax2,order = seg1['household_income'].value_counts().index ) #array, top subplot
            ax2 = fig.add_subplot(gs[1:4,2:4])
            ax2 = df[v].value_counts().plot(kind='pie',autopct='%.f%%',)
            ax2.set_xlabel(v)
            ax2.set_ylabel(None)
            
        
        
        if(v=='age_group_psnx'):
#            # row 5 and 6
            ax3 = fig.add_subplot(gs[4:6,2:4])
            sns.countplot(df.loc[:,v], ax=ax3,order = df[v].value_counts().index ) #array, top subplot
            vPercent(ax3,df)
            ax3.set_xlabel(v)
            ax3.set_ylabel(None)
        
            
#        if(v=='number_of_children'):
#            ax3 = fig.add_subplot(gs[4,2])
#            sns.countplot(df.loc[:,v], ax=ax3, ) #array, top subplot
#            vPercent(ax3,df)
#            plt.xlabel(v)
#            plt.ylabel(None)
        
        if(v=='marital_status'):
            ax3 = fig.add_subplot(gs[4,0])
            sns.countplot(y=df.loc[:,v], ax=ax3, ) #array, top subplot
            hPercent(ax3,df)
            plt.xlabel(v)
            plt.ylabel(None)
            
                
        if(v=='children'):
            ax3 = fig.add_subplot(gs[4,1])
            sns.countplot(df.loc[:,v], ax=ax3,palette={'Y':'limegreen','N':'#FA8072'} ) #array, top subplot
            vPercent(ax3,df)
            plt.xlabel(v)
            plt.ylabel(None)
        
        if(v=='household_size'):
            ax3 = fig.add_subplot(gs[5,0])
            sns.countplot(df.loc[:,v], ax=ax3,) #array, top subplot
            vPercent(ax3,df)
            plt.xlabel(v)
            plt.ylabel(None)
            
            
        if(v=='house_owner'):
            # row 5 and 6
            ax3 = fig.add_subplot(gs[5,1])
            sns.countplot(df.loc[:,v], ax=ax3,order = df[v].value_counts().index ) #array, top subplot
            vPercent(ax3,df)
            plt.xlabel(v)
            plt.ylabel(None)
        
       
#        if(v=='age_range2'):
#            ax4 = fig.add_subplot(gs[1:3,0:2])
#            sns.countplot(df.loc[:,v], ax=ax4,  ) #array, top subplot
#            vPercent(ax4,df)
#            plt.xlabel(v)
#            plt.ylabel(None)
#            
#        if(v=='grand_children'):
#            ax4 = fig.add_subplot(gs[4,0])
#            sns.countplot(df.loc[:,v], ax=ax4, ) #array, top subplot
#            vPercent(ax4,df)
#            plt.xlabel(v)
#            plt.ylabel(None)
#            
#        if(v=='interests__sports'):
#            ax4 = fig.add_subplot(gs[4,1])
#            sns.countplot(df.loc[:,v], ax=ax4, ) #array, top subplot
#            vPercent(ax4,df)
#            plt.xlabel(v)
#            plt.ylabel(None)
#            
#        if(v=='interests_travel'):
#            ax4 = fig.add_subplot(gs[5,0])
#            sns.countplot(df.loc[:,v], ax=ax4, ) #array, top subplot
#            vPercent(ax4,df)
##            plt.title(Marital_Status)
#            plt.xlabel(v)
#            plt.ylabel(None)
     
    plt.show()

