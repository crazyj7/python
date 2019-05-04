'''

https://www.kaggle.com/spscientist/students-performance-in-exams

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
sns.set(style="darkgrid", palette="bright", font_scale=1.5)


df = pd.read_csv("./StudentsPerformance.csv")
print(df.head())

print(df.describe())

sns.pairplot(df[['math score', 'reading score', 'writing score']], height=4)
plt.show()

def average_score(dt):
    return (dt['math score'] + dt['reading score'] + dt['writing score']) / 3

df['average score'] = df.apply(average_score, axis=1)

df.head()


sns.catplot(x='lunch', y='math score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('math')
plt.show()

sns.catplot(x='lunch', y='reading score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('reading')
plt.show()

sns.catplot(x='lunch', y='writing score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('writing')
plt.show()


sns.catplot(x='lunch', y='average score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('average')
plt.show()

sns.catplot(x='test preparation course', y='average score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('average')
plt.show()

sns.catplot(x='parental level of education', y='average score', kind='boxen', data=df, height=14)
plt.title('average')
plt.legend(loc='lower right')
plt.show()

