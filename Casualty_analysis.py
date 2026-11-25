# importing libraries for the data analysis

#Test for github commit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
) 
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



#---------------------------------------------------------------------------------------------
# 1:  DATA LOADING
# setting for the seaborn
sns.set_theme(style='whitegrid')

# Load the dataset, use low memory false because of the size of the data set
df = pd.read_csv('dft-road-casualty-statistics-casualty-2024.csv', low_memory=False)


print(100*"=")
# print the cols and rows
print(f'Dataset Shape: {df.shape}') 
print('\nColumn Names:')
# print all cols
print(df.columns.tolist())


# Quickly assess at the target variable: casualty_severity which can be:1=Fatal, 2=Serious, 3=Slight

print(100*"=")
# use an if to prevent error if the name changes dropna
if 'casualty_severity' in df.columns:
  print('\nCasualty Severity count:')
  print(df['casualty_severity'].value_counts(dropna=False))
#---------------------------------------------------------------------------------------------
# 2: DATA CLEANING
print(100*"=")

# Data cleaning, checking for -1 and 99 in important colums 
# Scan all 
print('All columns possible missing entries with -1:')
print((df == -1).sum())
print('All columns possible missing entries with 99:')
print((df == 99).sum())

# specific columns to check for -1 and 99
cols_to_monitor = ['age_of_casualty', 'age_band_of_casualty','sex_of_casualty', 'casualty_type','casualty_class','pedestrian_location', 'pedestrian_movement']

print('possible missing entries with -1:')
print((df[cols_to_monitor] == -1).sum())
print('possible missing entries with 99:')
print((df[cols_to_monitor] == 99).sum())

# 1 = Male, 2= Female, 9 = Unknown
# checking for 9 aswell
if 'sex_of_casualty' in df.columns:
    print('sex of casualty equal to 9 (Unknown):')
    print((df['sex_of_casualty'] == 9).sum())
    
print(100*"=")
# Replacement lines - np.nan is the NaN in numpy
cols_to_replace_missing = ['age_of_casualty', 'age_band_of_casualty','sex_of_casualty', 'casualty_type','casualty_class','pedestrian_location', 'pedestrian_movement']
df[cols_to_replace_missing] = df[cols_to_replace_missing].replace(-1, np.nan)

if 'sex_of_casualty' in df.columns:
  df['sex_of_casualty'] = df['sex_of_casualty'].replace(9, np.nan)

# Make verification of changes
print(100*"=")
print('After cleaning: missing entries (np.nan):')
print(df[cols_to_replace_missing].isna().sum())
print('Any remaining -1 in key columns:')
print((df[cols_to_replace_missing] == -1).sum())
print("Any remaining 99 in 'sex_of_casualty' columns:")
print((df['sex_of_casualty'] == 99).sum())
print(100*"=")

# Analysis of age of casualty
# use .value_counts() and .sort_index()
age_counts = df['age_of_casualty'].value_counts().sort_index()

print("Age : Number of Casualties")
print(50*"=")
# Read the entire list
print(age_counts.to_string())


#------------------------------------------------------------------------------
# 3 SEVERITY ANALYSIS

print("="*100)
print("Severity analysis of discreet variables")
print("="*100)

# Define a binary classification between fata and serious (1 or 2) vs slight (3)
# For more simplistic analysis of plots to generate percentages, fatal serious grouped, slight 
# The fataities are 1 to 2%, models will also struggle with a 3 way class with a very minor component
# Creates a new column with boolean values

if 'casualty_severity' in df.columns:
    df['severe_outcome'] = df['casualty_severity'].isin([1, 2])

# Examine possible discreet candidates
candidate_vars = ['casualty_type', 'casualty_class', 'age_of_casualty','age_band_of_casualty','sex_of_casualty', 'pedestrian_location', 'pedestrian_movement']

# produce a summary statistics for each of the candidate variables, sum considers True as 1 and False as 0 
for col in candidate_vars:
  if col in df.columns:
    summary = (
        df.groupby(col,dropna=False, observed=False)['severe_outcome']
        .agg(
            total_casualties='count',
            severe_casualties='sum',
        ).reset_index()
    )
    # Create a new percentage column of severe casualties 
    summary['%_severe'] = (
    (summary['severe_casualties'] / summary['total_casualties']) * 100).round(2)
    print(f'Severity analysisfor {col} ---')
    print(summary)

#----------------------------------------------------------------------------------------
# 4 FEATURE ENGINEERING

print("="*100)
print("Converting code from dictionary for severity analysis")
print("="*100)

# feature engineering
# For Casualty type this is the dictionary of the codes in the guide
# Dictionaries 
# I.a - Casualty type dictionary
category_definitions = {
    0: 'Pedestrians',
    1: 'Cyclists',
    2: 'Motorcyclists', 3: 'Motorcyclists', 4: 'Motorcyclists', 5: 'Motorcyclists', 
    22: 'Motorcyclists',23: 'Motorcyclists', 97: 'Motorcyclists',
    8: 'Car Occupants', 9: 'Car Occupants',
    10: 'Bus / Minibus / Coach', 11: 'Bus / Minibus / Coach',
    19: 'Goods Vehicles', 20: 'Goods Vehicles', 21: 'Goods Vehicles',
    90: 'Other / Unknown Vehicle Occupant', 99: 'Other / Unknown Vehicle Occupant',
}

# I.b - protection groups (VRU (vulnerable road users) vs PO (protected occupants)) 
vru_codes = [0,1,2,3,4,5,22,23,97]
po_codes = [8, 9, 10, 11, 19, 20, 21]
# protection groups
protection_map = {}

for code in vru_codes:
    protection_map[code] = 'VRU'
for code in po_codes:
    protection_map[code] = 'PO'


# II - casualty_class_mapping dictionary
casualty_class_mapping = {
    1:'Driver or rider',
    2:'Passenger',
    3:'Pedestrian'}

# III Age Granular 
# Bins chosen: -1 to 15, 15 to 24, 24 to 59, 59 to 74, 74 to 120
five_bins = [-1, 15, 24, 59, 74, 120]
labels_five = ['0-15 (child)',
               '16-24 (young)',
               '25-59 (adult)', 
               '60-74 (older)',
               '75+ (elderly)'
              ]
df['five_age_groups'] = (
    pd.cut(df['age_of_casualty'], bins=five_bins, labels=labels_five)
    .astype(str)
    .replace('nan', 'Unknown')
)

# IV - Age band of casualty dictionary
category_age_band_of_casualty = {
    1:'0-5',
    2:'6-10',
    3:'11-15',
    4:'16-20',
    5:'21-25',
    6:'26-35',
    7:'36-45',
    8:'46-55',
    9:'56-65',
    10:'66-75',
    11:'Over 75'
}

# V - Sex mapping dictionary
sex_mapping = {
    1: 'Male', 
    2: 'Female'
    }

# VI - Pedestrian location dictionary
category_def_ped_location = {
    0:'Not a pedestrian',
    1:'Crossing on pedestrian crossing facility',
    2:'Crossing in zig-zag approach lines',
    3:'Crossing in zig-zag exit lines',
    4:'Crossing elsewhere within 50m. of pedestrian crossing',
    5:'In carriageway, crossing elsewhere',
    6:'On footway or verge',
    7:'On refuge, central island or central reservation',
    8:'In centre of carriageway - not on refuge, island or central reservation',
    9:'In carriageway, not crossing',
    10:'Unknown or other'
}
#mbpsv = masked by parked or stationary vehicle

# VII - Pedestrian movement dictionary 
category_def_ped_movement = {
    0: "Not a pedestrian",
    1: "Crossing from driver's nearside",
    2: "Crossing from nearside- (mbpsv)",
    3: "Crossing from driver's offside",
    4:"Crossing from offside - (mbpsv)",
    5:"In carriageway,stationary-not crossing(standing or playing)",
    6:"In carriageway,stationary-not crossing(standing or playing) (mbpsv)",
    7:"Walking along in carriageway, facing traffic",
    8:"Walking along in carriageway, back to traffic",
    9:"Unknown or other"
}

#  Mapping of the dictionaries (I-VII)
categories = [(category_definitions, 'casualty_type'), 
              (category_age_band_of_casualty, 'age_band_of_casualty'), 
              (sex_mapping, 'sex_of_casualty'), 
              (casualty_class_mapping, 'casualty_class'), 
              (category_def_ped_location, 'pedestrian_location'),
              (category_def_ped_movement, 'pedestrian_movement'),
            ]

for mapping_dict, col_name in categories:
    df[col_name + "_mapped"] = df[col_name].map(mapping_dict).fillna('Unknown')

df['casualty_type_mapped_pro'] = (
    df['casualty_type'].map(protection_map).fillna('Other/Unknown')
)

# Outputting the mapped values
# Verify
print(100*"=")
print('casualty_type all casualty count: \n')
print(df['casualty_type_mapped'].value_counts())
print(25*"=") 
print('casualty_type_protected all casualty count: \n')
print(df['casualty_type_mapped_pro'].value_counts())
print(25*"=") 
print('age band of casualty all casualty count: \n')
print(df['age_band_of_casualty_mapped'].value_counts())
print(25*"=") 
print('sex of casualty all casualty count: \n')
print(df['sex_of_casualty_mapped'].value_counts())
print(25*"=") 
print('casualty class all casualty count: \n')
print(df['casualty_class_mapped'].value_counts())
print(25*"=") 
print('pedestrian location all casualty count: \n')
print(df['pedestrian_location_mapped'].value_counts())
print(25*"=") 
print('pedestrian movement all casualty count: \n')
print(df['pedestrian_movement_mapped'].value_counts()) 
print(25*"=") 
print('five age groups all casualty count: \n')
print(df['five_age_groups'].value_counts())

#------------------------------------------------------------------------------------------------------

# 5 SEVERITY ANALYSIS 

#severity for the word mapped and engineered features
print("="*100)

engineered_features = [
    'casualty_type_mapped',  # Grouped into 7 modes (Pedestrians, Cyclists, etc.)
    'casualty_type_mapped_pro', # Grouped into the protected and vulnerable 
    'sex_of_casualty_mapped', # Casualty sex with words
    'casualty_class_mapped', # Casualty class with words
    'pedestrian_location_mapped', # pedestrians position labelled
    'pedestrian_movement_mapped', # pedestrian movement labelled
    'five_age_groups', # age groups divided into five groups
    'age_band_of_casualty_mapped', # age bands with smaller divisions
]

# Just repeated from higher up in the code
df['severe_outcome'] = df['casualty_severity'].isin([1, 2])


# 
# severity analysis
for col in engineered_features:
    if col in df.columns: # defensive programming check, ensures the column exist in df
        summary = (
            df.groupby(col, observed=False)['severe_outcome']
            .agg(total_casualties='count', 
                 severe_casualties='sum'
                )
        .reset_index() # regular dataframe
        ) 
        # Group by the unique categories in that columns (pedestrian, cyclist ect)
        # focused aggregation for the boolean column
        # two metrics are calculated simultaneously 'count', 'sum'
        
        # filter out any category that has 0 casualties to avoid /0 errror. .copy to prevent a warning 
        summary = summary[summary['total_casualties'] > 0].copy()
        # will skip look if there is a column with no data
        if summary.empty:
            continue

        # Calculate percentage of severe casualties
        summary['%_severe'] = (
        (summary['severe_casualties'] / summary['total_casualties']) * 100
        ).round(2)
        #  organises in a sorted manner
        summary = summary.sort_values(by='%_severe', ascending=False)

        #ascending=False : places the highest-risk categories motorcyles ext. at the tope
        # prints the numerical summary of these severity
        print(f'\n----Severity Analysis for: {col}---')
        print(summary.to_string(index=False))

        # -------------------------------------------------------------
        # Plotting directly inside this loop 
        # Adjust the figure height which is proportial to catagories by change the height to be 
        # proportional to the number of categories as they are horizontal plots
        fig_height = max(3.5, len(summary) * 0.5)
        fig, axis = plt.subplots(figsize=(10, fig_height))

        # Horizontal bar plot 
        sns.barplot(
            data=summary, # plots the summary
            x='%_severe', # percentage of severity on the x-axis
            y=col, # catagory names on the y-axis
            palette='YlOrRd_r',
            hue=col,  # unique bar colours
            legend=False,
            ax = axis # drawn on axes
        )

        # Annotate each bar with the exact percentage
        for p in axis.patches: #loops all the bars
            width = p.get_width()
            if not np.isnan(width) and width > 0: 
                axis.annotate(
                f' {width:.1f}%', # one decimal place eg. 20.1%
                (width, p.get_y() + p.get_height() / 2),
                va='center', # text center
                ha='left', 
                fontsize=9,
                fontweight='bold',
            )
        # Title - clean up the title modified by mapping, gets rid of underscores replace with empty
        clean_name = col.replace('_mapped', '').replace('_', ' ').title()
        axis.set_title(
            f'Severe Casualty Rate (%) by {clean_name}', 
            fontsize=14, 
            fontweight='bold',
            pad=10
            )
        axis.set_xlabel('Severe Casualty Rate (%)')
        axis.set_ylabel(clean_name)
        axis.set_xlim(0, max(summary['%_severe']) * 1.2) # extend the right edge of x-axis
        
        plt.tight_layout() # automatically adjusts the margins, axis titles and tick marks

        # Save to disk first
        plt.savefig(f'severity_plot_{col}.png', dpi=300, bbox_inches='tight')

        # Display plot
        plt.show()

          
#--------------------------------------------------------------------------------------------

# 5 STATISTICAL TESTS 
# Statisitcal test (Xi^2 and cramer-V on candidate columns engineered
# same as the engineered features
test_features = [
    'casualty_type_mapped',  # Grouped into 7 modes (Pedestrians, Cyclists, etc.)
    'casualty_type_mapped_pro', # Grouped into the protected and vulnerable 
    'sex_of_casualty_mapped', # Casualty sex with words
    'casualty_class_mapped', # Casualty class with words
    'pedestrian_location_mapped', # pedestrians position labelled
    'pedestrian_movement_mapped', # pedestrian movement labelled
    'five_age_groups', # age groups divided into five groups
    'age_band_of_casualty_mapped', # age bands with smaller divisions
]

# empty list that as the for each pass of the for loops over the candidate features
# Xi^2, p-value, degrees of freedom, Cramer's V
# These will be appended as dictionaries into the list
stats_results = []
# creates a contingency table - rows catagories and columns True or False
for col in test_features:
    
  if col in df.columns:
    # ||Xi^2||
    # Build a two-way contingency table (frequency grid) with the df[featue] and df[severe_outome] as argument.
    # Rows are the distinct category feature (pedestrian, motocylist ect.)
    # Columns are the binary outcome
    # The count of casualties for that relation
    contingency_tab = pd.crosstab(df[col], df['severe_outcome'])
    # chi2 how much is differes from expected severity if completely independent:
    #  Xi^2 = SUM((oberved -expected)^2/expected
    # p is p-value: the  probabilty that this pattern ocurred by chance
    # dof number of independent frquency values that can change ( rows-1) x (column -1)
    # expected: frequency that would occur under complete independence E = (row total)x(column total)/(grand total)
    chi2, p, dof, expected = chi2_contingency(contingency_tab)
    n = contingency_tab.sum().sum() # Sums all the entries in the contingency table
    # ||Cramer's_V ||
    # The Xi^2 is inflated because of the dataset size, Crame's V indicates strength
    # V = sqrt [  Xi^2/(n x min(r-1, k-1))] , sqrt to become linear
    # r is the number of row k s the number of columns, n os the sum
    # Xi^2max = n x min(r-1, k-1)
    min_dim = min(contingency_tab.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    # 0.00 -0.10 is a weak association, 0.1-0.25 moderate association, > 0.25 strong assoication  
    # append the results to the stats summary list as one dictionary for each feature
    stats_results.append({
        'Feature': col,
        'Chi2_Statistic': round(chi2, 2),
        'p_value': '< 0.001' if p < 0.001 else f'{p:.4f}', # four decimal places
        'Degrees_of_Freedom': dof,
        'Cramers_V': round(cramers_v, 4),
    })
    # Residuals for severe casualties (column True / 1)
    residuals = (contingency_tab - expected) / (expected**0.5)
    # create a new dataframe breakdown: the residuals for each feature, +2 overrepresenting severe, -2 under representing severe injuries
    breakdown = pd.DataFrame({
            'Total_Casualties': contingency_tab.sum(axis=1), # axis 1 sum across the columns both severe and non severe
            'Severe_Observed': contingency_tab[True], #  column of severe = True
            'Severe_Expected': expected[:, 1].round(1), # slices the row to get severe expected if zero effect of catagory
            'Standardized_Residual': residuals[True].round(2), # standardised residuals  
            }).sort_values(by='Standardized_Residual', ascending=False) # high to low sorting
    print(f'=======Breakdown for {col}:==========')
    print('\n')
    print(breakdown)

stats_df = pd.DataFrame(stats_results).sort_values(by='Cramers_V', ascending=False)
print(100*"=")
print('=====Quantitative Association Ranking======')
print(stats_df.to_string(index=False))


#-----------------------------------------------------------------------------------------------------
# 6 THREE WAY ANALYSIS 
# Three way analysis of age_band of casualty, casualty type and severity
# exploratory 3-way heatmap and interaction curve.

# The requires an contingency table and annotated heatmap

# repeat the severe_outcome calculation from above, TRUE if 1,2 and FALSE 3

if 'casualty_severity' in df.columns:
    df['severe_outcome'] = df['casualty_severity'].isin([1, 2])

# New age band lists repeated
age_bands = [
    '0-5',
    '6-10',
    '11-15',
    '16-20',
    '21-25',
    '26-35',
    '36-45',
    '46-55',
    '56-65',
    '66-75',
    'Over 75',
]

# Here remove unallocated values for a cleaner modelling
# 'age_band_of_casualty_mapped', casualty_type_mapped defined above, having already had -1 removed.
analysis_df = df[
    (df['age_band_of_casualty_mapped'].isin(age_bands)) # first filter that is in category_age_band_of_casualty dictionary values and not unknown
     & (
        ~df['casualty_type_mapped'].isin(['Unknown', 'Other / Unknown Vehicle Occupant'])) # use the inverse ~ to filter the 'Unknown' 'Other / Unknown Vehicle Occupant' 
    ].copy()

# A 3-way cross tabulation to make a 2D heat map representation of the data
# pivot table, where the index is the age band and the columns are the casualty type, agg function is 'mean'
pivot_rate = analysis_df.pivot_table(
    index='age_band_of_casualty_mapped', # Age parameter
    columns='casualty_type_mapped', # Casualty type degrees of freedom
    values='severe_outcome', # Severe outcome (True = 1,2) (False =3)
    aggfunc='mean',
    observed=False
)*100

pivot_rate = pivot_rate.reindex(age_bands)


print('=======Severe casualty rate (%) by Age Band (5 years) and Casualty Type 10 df=======')
print(pivot_rate.round(2))

# Visualising the heatmap from the pivot table

plt.figure(figsize=(12, 9)) # This us a seaborn heatmap
sns.heatmap(
    pivot_rate,
    annot=True,
    fmt='.1f',
    cmap='YlOrRd', # same colour as the other grophs
    cbar_kws={'label': 'Severe casualty rate (%)'},
    linewidths=0.8,
)

plt.title(
    'Severe casualty rate (%) by Age band (5 years) and Casualty type 10 df',
    fontsize=14,
    fontweight='bold',
    pad=12,
)

plt.xlabel('Casualty Type (Mode of Transport or Pedestrian)', fontsize=12, fontweight='bold')
plt.ylabel('Age Band of Casualty', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('task1b_interaction_heatmap.png', dpi=300, bbox_inches='tight') # print the fig
plt.show()

# visualisation multi-line interaction profiles on a graph

main_modes = [
    'Pedestrians',
    'Cyclists',
    'Motorcyclists',
    'Car Occupants',
    'Bus / Minibus / Coach',
]

# list comprehension to get all modes from the pivot_rate
# This checks the actual modes present in the pivot_rate.column against the main_modes and extracts the subset
available_modes = [mode for mode in main_modes if mode in pivot_rate.columns]

plt.figure(figsize=(12, 9))
palette = sns.color_palette('tab10', len(available_modes))  # dynamic if the number of modes it can adjust size

for i, mode in enumerate(available_modes):  # enumerate creates an index and mode tuple pair for every mode
    # Use a standard plt
    plt.plot(
      pivot_rate.index,
      pivot_rate[mode],
      marker='X',
      linewidth=2.5,
      label=mode,
      color=palette[i],
  )

plt.title(
    'Figure 2: Age-Severity Trajectories Across Transport Modes',
    fontsize=13,
    fontweight='bold',
    pad=12,
)

plt.xlabel('Age Band of Casualty', fontsize=11, fontweight='bold')
plt.ylabel('Severe Casualty Rate (%)', fontsize=11, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(title='Transport Mode', frameon=True)
plt.tight_layout()
plt.savefig('task1b_severity_trajectories.png', dpi=300, bbox_inches='tight')
plt.show()

# SECTION 2 - MACHINE LEARNING MODELS
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# 7 MACHINE LEARNING MODELLING of age and casualty type

# Random Forest and Decision Trees are used to provide a tree based model
# The decision tree is white box model to visualise (at reasonable depths) the tree

print("=" * 60)
print("---Predictive Modelling Age vs Casualty Type---")
print("---DecisionTree---")
print("---RandomForest---")
print("---logistic regression---")
print("=" * 60)

# extrat the valid age bands directly from the mapping dictionary
# In full to emphasize the baseline index
age_bands = [
    '36-45', # baseline (index 0). For the drop first.
    '0-5', 
    '6-10',
    '11-15',
    '16-20',
    '21-25',
    '26-35',
    '46-55',
    '56-65',
    '66-75',
    'Over 75',
]

casualty_categories = [
    'Car Occupants',  # baseline (index 0). For the drop first.
    'Pedestrians',
    'Cyclists',
    'Motorcyclists',
    'Bus / Minibus / Coach',
    'Goods Vehicles',
]

# Filter model_df using dictionary values to only obtain these
model_df = df[
    (df['age_band_of_casualty_mapped'].isin(age_bands))
    & (~df['casualty_type_mapped'].isin( ['Unknown', 'Other / Unknown Vehicle Occupant'])
    )
].copy()

# Target variable repeat of severity binary classification: True/1, Fatal/serious, False/0, slight
model_df['severe_outcome'] = (model_df['casualty_severity'].isin([1, 2]).astype(int))

# Rename the mapped columns to make clean titles from the mapped versions with.rename
model_df = model_df.rename(
    columns={
        'casualty_type_mapped': 'Casualty Type:',
        'age_band_of_casualty_mapped': 'Age Band:',
    }
)
# To prevent the alphabetical ordering categorical using the above lists. 
model_df['Casualty Type:'] = pd.Categorical(
                      model_df['Casualty Type:'], categories=casualty_categories
                      )

model_df['Age Band:'] = pd.Categorical(
                      model_df['Age Band:'], categories=age_bands
                      )


# These are the features to be one hot coded into 1 or 0
features = ['Age Band:', 'Casualty Type:']

# One-hot encode both categorical columns, the first is dropped here, car occupants and ages 36-45 10 year group
X = pd.get_dummies(model_df[features], drop_first=True)
y = model_df['severe_outcome']

# splitting data according to the brief, 80/20 where 80 is the training and 20 is testing
splits = {'80/20 Split': 0.20,'70/30 Split': 0.30,'60/40 Split': 0.40}

# Instantiate Decision and Random forest and logistic regression classifiers as models dictionary
# max_depth limits tree growth to prevent extreme overfitting in Decision Tree


# Models
# Binary decision tree classifier.
# ---Hyperparameters-----:
# class_weight='balanced': severe class imbalance, fatal serious 15-20%, without slight is overpredicted
# weight = n(sample)/(n(class)xn(sample,class). Its better to reduce the False negatives at expense of false positive
# Depth of 5 is chosen to result in 2^5 terminal leaves, can capture complex interacrtions
# n_estimators of 100 this is reasonable number of indvidual trees
# max_iter=1000, guarnatees gradient covergence
# deterministic - (42) 


models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced' ),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
}

# Evaluate models across the 80/20, 70/30, 60/40 splits
results = []
# Outer loop for the splits
for split_name, test_size_val in splits.items():
    # Split the dataset
            
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size_val, # from split dictionary
        random_state=42, 
        stratify=y  # Maintains a balanced severity ratios in train & test sets
    )
    # Cycle through the three decision tree and random forest and logistic models
    # Trains each of the classification algorithms
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train) 
        
        # Predict on test set
        y_pred = model.predict(X_test)  # hard binary classification 0 = slight, 1 = severe/fatal, standard 0.5 threshold
        y_prob = model.predict_proba(X_test)[:, 1] # soft continuous probabilities 0.0-1.0
        # 2d array col 0 prob of class 0, col 1 is prob of class 1. The slice [:,1] probabilty an incident is severe
        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred) 
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        # key: True positive (TP) (a correct severe), True negative (TN) (a correct slight), 
        # False positive (FP) (a false severe), False negative (FN) (a false slight)
        # test results
        results.append({
            'Split': split_name,
            'Classifier': model_name,
            'Accuracy (%)': round(acc * 100, 2), # overall correct predictions (true severe, true slight (TP + TN) / (TP + TN + FP + FN)
            'Precision': round(prec, 4), # only true severe (TP / TP + FP)
            'Recall': round(rec, 4), # the sensitivity to find the severe casualties (! a priority when missing)
            'F1-Score': round(f1, 4), # 2 * (precision x recall/precision + recall), balance of false alarms against severe incidents
            'ROC-AUC': round(auc,4) # evaluates how the model ranks severe to slight
        })

# Display results table 
# accumularte results list into a comparison dataframe
comparison_df = pd.DataFrame(results)
print("===Classifier performance comparisons===")
print(comparison_df.to_string(index=False))

#---------------------------------------------------------------------------------------------
# 8 DEEPER DIAGNOSITCS OF ML Models

# I: Random Forest model importances
# Without dropping the first
# Only 80/20 split
X_full = pd.get_dummies(model_df[features], drop_first=False)

# Quick 80/20 split for feature analysis
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,y, test_size=0.2, random_state=42, stratify=y
)


# Random Fores Feature importances
rf_diagnostic = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'
)

# train model
rf_diagnostic.fit(X_train_full, y_train_full)
# Data frame of feature importance
feature_imp_df = (
    pd.DataFrame({'Feature': X_full.columns, 'Importance': rf_diagnostic.feature_importances_})
    .sort_values(by='Importance', ascending=False)
    .reset_index(drop=True)
)
# decending greatest features
print('\n===The decending greatest 10 RandomForest feature importances===')
print(feature_imp_df.head(10).to_string(index=False))

print('\n===The acending lowest 10 RandomForest feature importances===')
print(feature_imp_df.sort_values('Importance', ascending=True).head(10).to_string(index=False))

# II: Logisitc Regression Odds Ratios
# logistic model 
lr_diagnostic = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
# Train model
lr_diagnostic.fit(X_train, y_train)

# Extract log-odds array
log_odds = lr_diagnostic.coef_[0]
# log (odds ratios) and the exp of (log_odds)
odds_ratios = pd.DataFrame({
    'Feature': X.columns,
    'Log-Odds': log_odds.round(4),
    'Odds Ratio': np.exp(log_odds).round(4),
    'Effect': np.where(log_odds > 0, 'Increases:(+)', 'Decreases:(-)'),
}).sort_values('Odds Ratio', ascending=False)

print('\n===The top 10 severity risk multipliers (greatest odds ratios) ===')
print(odds_ratios.sort_values('Odds Ratio', ascending=False)
      .head(10).to_string(index=False))

print('\n=== Top 10 protective factors (lowest odds ratios / negative log-odds) ===')
print(odds_ratios.sort_values('Odds Ratio', ascending=True)
      .head(10).to_string(index=False))

# III: Decision Tree Confusion Matrix - tradeoff of precision and recall
dt_5 = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
dt_5.fit(X_train, y_train)
y_pred_dt = dt_5.predict(X_test)

cm_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred_dt),
    index=['Actual Slight (0)', 'Actual Severe (1)'],
    columns=['Predicted Slight (0)', 'Predicted Severe (1)'],
)

print('\n===The decision tree Confusion Matrix (CM) (80/20 Test Set)===')
print(cm_df)

#---------------------------------------------------------------
# 9 VISUALISATION OF DECISION TREE 

# Use max_depth=3 (3 levels, therefore 2^3=8 leaves. for visualisation so the diagram is clear and readable
# Instantiates a classification tree object from sci-kit-learn
# Class_weight='balanced' offsets the 80:20 casualty imbalance in dataset. Weigths the loss function inversely

# Instantiates binary decision tree classifier
dt_visual = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Train the tree on training partition 
# maximises information gain (Gini reduction at each node)
dt_visual.fit(X_train, y_train)

#Plotting a figure 
plt.figure(figsize=(20, 10)) # 20 in wide and 10 inches tall big enough for the tree
# matplot lib from skikit-learn
plot_tree(
    dt_visual,
    feature_names=X.columns.tolist(), # applies a meaningful feature names which are used to separate
    class_names=['Slight', 'Severe (Fatal/Serious)'], # maps the target class targets 0,1> slight, severe(fatal/serious)
    filled=True,  # Colors nodes according to majority class, color intensity indicates concentration
    rounded=True,  # Rounded box edges rather than sharp
    fontsize=12,  
    impurity=True,  # Shows Gini impurity at each node
    proportion=True,  # Displays sample percentage (0.0-1.00 rather than raw counts, for easier interpretatoin
)

plt.title(
    'Decision Tree Splitting Logic: Age Band vs Casualty Type',
    fontsize=15,
    fontweight='bold',
    pad=15, # prevents overlap
)
plt.tight_layout() # adjusts subplot margins to eliminate dead space
plt.savefig('task1b_decision_tree_extracted.png', dpi=300, bbox_inches='tight')
plt.show()
# Translates the visual geometric tree structure into a clear readable (by human) textwith IF-THEN logic 
# explaining the logic of the tree.

tree_rules = export_text(
    dt_visual, # tree model after going throught training pipeline at depth 3
    feature_names=X.columns.tolist(), # replaces X[0] X[1] with meaninful feature names
    show_weights=True,  # Shows class distribution in each leaf of slight vs severe
)

print('\n'+'='*60)
print('====Extracted Decision Tree Rules (IF-THEN Logic)====')
print('='*60)
print(tree_rules)

# Optionally save rules to a text file for your code appendix
with open('task1b_tree_rules.txt', 'w') as f:
  f.write(tree_rules)


#---------------------------------------------------------------------------------
# 10 VARIABLE DEPTH OF DECISION TREE 
# Iterate throught different depths for decisiion tree and random forest, 1-20

depths = [1,2,3,5,10,20]
# 0.2 testing and 0.8 training split as used for the other analysis

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size= 0.2, 
        random_state=42, 
        stratify=y  # Maintains a balanced severity ratios in train & test sets
        )

results = []  # Initialize outside the loop to collect all the depth results

#Other hyper parameters are maintained
for dep_ in depths:
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=dep_, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=dep_, random_state=42, class_weight='balanced' )
        }
    
    # Split the dataset
    
    # Cycle through the decision tree and random forest and logistic models
    
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        #I.Evaluate on the training set (memorise)
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        acc_train = accuracy_score(y_train, y_train_pred)
        auc_train = roc_auc_score(y_train, y_train_prob)


        #II.Evaluate on the test set (general, new data)
        
        # Predict on test set
        y_pred = model.predict(X_test) # hard prediction
        y_prob = model.predict_proba(X_test)[:, 1] # soft probabiltiies 0-1
        
        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Quantified overfitting gaps (train-test)
        acc_gap = round((acc_train - acc) * 100, 2)
        auc_gap = round(auc_train - auc, 4)
        
        results.append({
            'Depth' : dep_,
            'Split': '80/20',
            'Classifier': model_name,
            'Train Accuracy (%)': round(acc_train * 100, 2),
            'Test Accuracy (%)': round(acc* 100, 2),
            'Accuracy Gap (%)': acc_gap,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Train AUC': round(auc_train, 4),
            'Test AUC': round(auc, 4),
            'AUC Gap': auc_gap,
            
        })

# Display results table
comparison_df = pd.DataFrame(results)
print("\n===DT and RF Classifier performance with depth===")
print(comparison_df.to_string(index=False))



