# Import necessary libraries
import pandas as pd

# Load Pokemon data from CSV files
pokemon = pd.read_csv('pokemon.csv')
abilities = pd.read_csv('abilities.csv')

# Remove irrelevant columns
pokemon = pokemon.drop(['#', 'Generation', 'Legendary'], axis=1)
abilities = abilities.drop(['id'], axis=1)

# Handle missing and duplicate data
pokemon = pokemon.drop_duplicates()
abilities = abilities.drop_duplicates()
pokemon = pokemon.dropna()
abilities = abilities.dropna()

# Convert data types to appropriate formats
pokemon['Type 2'] = pokemon['Type 2'].fillna('')
pokemon[['Type 1', 'Type 2']] = pokemon[['Type 1', 'Type 2']].astype('category')
abilities['ability'] = abilities['ability'].astype('category')

# Calculate summary statistics for key variables
print(pokemon[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].describe())

# Create scatterplot matrix to explore relationships between variables
pd.plotting.scatter_matrix(pokemon[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']])
plt.show()

# Create bar chart to show the distribution of Pokemon types
pokemon['Type 1'].value_counts().plot(kind='bar')
plt.show()

# Create histogram to show the distribution of Pokemon attack power
pokemon['Attack'].plot(kind='hist')
plt.show()

# Identify any outliers in the dataset
pokemon.boxplot(column=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'])
plt.show()

# Create scatterplot to show the relationship between attack and defense power
plt.scatter(pokemon['Attack'], pokemon['Defense'])
plt.xlabel('Attack Power')
plt.ylabel('Defense Power')
plt.show()

# Create heatmap to show the correlation between different attributes
import seaborn as sns
corr_matrix = pokemon.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Create stacked bar chart to show the proportion of each Pokemon type that has a secondary type
type1 = pokemon['Type 1']
type2 = pokemon['Type 2']
both_types = type1 + '-' + type2
both_types.value_counts().plot(kind='bar', stacked=True)
plt.show()

# Statistical Analysis

# Mean and Median of Attack and Defense
mean_attack = pokemon_df['Attack'].mean()
mean_defense = pokemon_df['Defense'].mean()
median_attack = pokemon_df['Attack'].median()
median_defense = pokemon_df['Defense'].median()

print("Mean Attack: {:.2f}".format(mean_attack))
print("Mean Defense: {:.2f}".format(mean_defense))
print("Median Attack: {}".format(median_attack))
print("Median Defense: {}".format(median_defense))

# Mode of Type 1 and Legendary
mode_type_1 = pokemon_df['Type 1'].mode()[0]
mode_legendary = pokemon_df['Legendary'].mode()[0]

print("Mode Type 1: {}".format(mode_type_1))
print("Mode Legendary: {}".format(mode_legendary))

# Standard Deviation of HP and Speed
std_hp = pokemon_df['HP'].std()
std_speed = pokemon_df['Speed'].std()

print("Standard Deviation HP: {:.2f}".format(std_hp))
print("Standard Deviation Speed: {:.2f}".format(std_speed))

# Correlation between variables
correlation = pokemon_df.corr()
print(correlation)
