import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Nikhil1.csv', lineterminator='\n')
print(df.head())
print(df.info())

#starting five genre
print(df['Genre'].head())

#all genre show
# print(df['Genre'])

#check duplicate value
print(df.duplicated().sum())

#check true false through
# print(df.duplicated())

#basic stats count,min,max,25,75,50,std
print(df.describe()) 

#date time foremate
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
print(df['Release_Date'].dtype)

#only year formate
df['Release_Date'] = df['Release_Date'].dt.year
print(df['Release_Date'].dtypes)
print(df.head())

#Remove column
cols = ['Overview', 'Original_language', 'Poster_Url']

# Check which of these columns exist in df
cols_to_drop = [col for col in cols if col in df.columns]

# Drop only those columns
df.drop(cols_to_drop, axis=1, inplace=True)

# Print remaining columns and first few rows
print("Remaining columns:")
print(df.columns)

print("\nData preview:")
print(df.head())

#add label in avg wise category 25% 50% 75%

# def catigorize_col(df,col,labels):
#     edges = [df[col].describe()['min'],
#              df[col].describe()['25%'],
#              df[col].describe()['50%'],
#              df[col].describe()['75%'],
#              df[col].describe()['max']]
#     df[col] = pd.cut(df[col],edges,labels = labels,duplicates='drop')
#     return df

# labels = ['not_popular','below_avg','average','popular']
# catigorize_col(df,'Vote_Average',labels)
# print(df['Vote_Average'].unique())


def catigorize_col(df, col, labels):
    # Convert column to numeric (in-place), ignore non-numeric issues
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN (non-convertible entries)
    df = df.dropna(subset=[col])

    # Get quantile edges
    q1 = df[col].quantile(0.25)
    q2 = df[col].quantile(0.50)
    q3 = df[col].quantile(0.75)
    min_val = df[col].min()
    max_val = df[col].max()

    edges = [min_val, q1, q2, q3, max_val]
    edges = sorted(set(edges))

    adjusted_labels = labels[:len(edges) - 1]

    df[col + '_category'] = pd.cut(
        df[col],
        bins=edges,
        labels=adjusted_labels,
        include_lowest=True,
        duplicates='drop'
    )

    return df

# Labels to apply
labels = ['not_popular', 'below_avg', 'average', 'popular']

# Call the function
df = catigorize_col(df, 'Vote_Average', labels)

# Output result
print(df[['Vote_Average', 'Vote_Average_category']].head())

#average number
print(df['Vote_Average'].value_counts())

#remove duplicate and none values

print(df.dropna(inplace = True))
print(df.isna().sum())

#ALL GENRE IN PARTICULAR COLUMN AND REMOVE WHITE SPACE

df['Genre'] = df['Genre'].str.split(',')
df = df.explode('Genre').reset_index(drop=True)
print(df.head())

#Casting column into category

df['Genre'] = df['Genre'].astype('category')
print(df['Genre'].dtype)

print(df.info())
print(df.nunique())


# Data Visualization

sns.set_style('whitegrid') 

# what is the most frequent genre of movies released on netflix?

print(df['Genre'].describe())

sns.catplot(y = 'Genre',data= df,kind='count',
            order=df['Genre'].value_counts().index,
            color='#4287f5')
plt.title('Genre column distribution')
print(plt.show())
            
# which has highest votes in vote avg column?

sns.catplot(y = 'Vote_Average',data= df,kind='count',
            order=df['Vote_Average'].value_counts().index,
            color='#4287f5')
plt.title('Vote_distribution')
print(plt.show())

#what movie got the highest popularity? what's its genre?

print(df[df['Popularity']==df['Popularity'].max()])

#what movie got the lowest popularity? what's its genre?

print(df[df['Popularity']==df['Popularity'].min()])

# which year has the most filmmed movies?

df['Release_Date'].hist()
plt.title('ReleaSE Date column distribution')
print(plt.show())