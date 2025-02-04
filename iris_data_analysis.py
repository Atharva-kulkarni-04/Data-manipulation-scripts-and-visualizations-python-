import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, header=None, names=column_names)

# Displaying  the first few rows of the iris datasey
print("First 5 rows of the dataset:")
print(iris_data.head())

# Data Manipulation Tasks
# 1.Filtering (Selecting the rows  where species is 'Iris-setosa'only)
setosa_data = iris_data[iris_data['species'] == 'Iris-setosa']
print("\nFiltered data for Iris-setosa:")
print(setosa_data)

# 2.Grouping (Grouping by species and calculate the mean)
grouped_data = iris_data.groupby('species').mean()
print("\nMean values for each species:")
print(grouped_data)

# 3.Aggregating (Count the number of occurrences of each species)
species_count = iris_data['species'].value_counts()
print("\nCount of each species:")
print(species_count)

# Data Visualization
# 1.Scattering plot of sepal length vs sepal width
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_data, x='sepal_length', y='sepal_width', hue='species', palette='Set1')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.grid()
plt.savefig('sepal_length_vs_sepal_width.png')  # Save the figure
plt.show()

# 2. Box plot of petal length by species
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_data, x='species', y='petal_length', palette='Set2')
plt.title('Box Plot of Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid()
plt.savefig('box_plot_petal_length.png')  # Save the figure
plt.show()