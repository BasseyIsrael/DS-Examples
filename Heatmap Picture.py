import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
urllink = 'https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv'
data_set = pd.read_csv(urllink)

pivotdata_set = data_set.pivot('country','year','lifeExp')
print (pivotdata_set)
plt.subplots(figsize=(40,40))
heatmap = sns.heatmap(pivotdata_set)
heatmap.get_figure().savefig('heatmap.png')