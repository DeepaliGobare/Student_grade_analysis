import pandas as pd
ds = pd.read_csv('student_grade_analysis_form.xlsx - in.csv')
ds.info()

# popping a column from our data frame as we are preparing our data

print(ds.head())

ds.pop('Timestamp')

print(ds.head())

ds.pop('Email Address')
ds.pop('Email')

print(ds.head())

# here we are renaming entries in the column such as No as No clubs, yes as Clubs as we have to make itemset of every entry such as No clubs must also be an item
#  and clubs would be another item in itemset.... then we will make every type of entry in the column a diffrent Column so that each type of entry can have chance to become a possible item in the itemSet

ds['Are you in any clubs?'].replace({'No' : 'No clubs', 'Yes' : 'Clubs'},inplace=True)
ds['Do you regularly ask doubts to teachers?'].replace({'No' : 'Not asking doubts', 'Yes' : 'Asks Doubts'}, inplace = True)
ds['Are you being raised by single parent?'].replace({'Yes' : 'Raised By Single Parent', 'No' : 'Raised By Both Parents'}, inplace = True)
print(ds.head())

# with the help of below function we can replace the cgpa with there specific category

def marksFiltering(x):
  if x > 9.5:
    return 'Above 9.5'
  elif x > 9:
    return '9 to 9.5'
  elif x > 8.5:
    return '8.5 to 9'
  else:
    return 'Below 8.5'


ds['What is your SGPA of previous semester (Semester 5)?'] = ds['What is your SGPA of previous semester (Semester 5)?'].apply(marksFiltering)

print(ds['What is your SGPA of previous semester (Semester 5)?'])

cgpa = ds.pop('What is your CGPA?')
ds['What is your aggregate attendance of previous semester?'].replace({'>75%' : '75', '<50%' : '50', '50 - 75 %' : '50to75'})
print(ds)


# here for all the 7 columns we will replace the columns with the specific entries it has.... and the values of those column would be in the form of 0/1... thats what we wanted as
# apriori algorithm will find out frequent items set using 1 & all.... 1 means it appears in this entry........ if we didnt have splitted the columns then what would have happened is
# for eg we had no clubs as No right... that will be 0... then apriori wouldnt have taken that Not being in club can also form a item in a itemset

ds = pd.concat([ds.drop('What is your SGPA of previous semester (Semester 5)?', axis=1), pd.get_dummies(ds['What is your SGPA of previous semester (Semester 5)?'])], axis = 1)
ds = pd.concat([ds.drop('What is your aggregate attendance of previous semester?', axis=1), pd.get_dummies(ds['What is your aggregate attendance of previous semester?'])], axis = 1)
ds = pd.concat([ds.drop('Are you in any clubs?', axis=1), pd.get_dummies(ds['Are you in any clubs?'])], axis = 1)
ds = pd.concat([ds.drop('Do you regularly ask doubts to teachers?', axis=1), pd.get_dummies(ds['Do you regularly ask doubts to teachers?'])], axis = 1)
ds = pd.concat([ds.drop("What is your family's annual income?", axis=1), pd.get_dummies(ds["What is your family's annual income?"])], axis = 1)
ds = pd.concat([ds.drop('While doing college assignments,', axis=1), pd.get_dummies(ds['While doing college assignments,'])], axis = 1)
ds = pd.concat([ds.drop('What is your parents occupation?', axis=1), pd.get_dummies(ds['What is your parents occupation?'])], axis = 1)
ds = pd.concat([ds.drop('Are you being raised by single parent?', axis=1), pd.get_dummies(ds['Are you being raised by single parent?'])], axis = 1)
ds.info()

# Print the first few rows of the DataFrame after one-hot encoding
print(ds.head())
print(ds)

from mlxtend.frequent_patterns import apriori, association_rules

# Find frequent itemsets using Apriori
frq_items = apriori(ds, min_support = 0.01, use_colnames = True,)

print(frq_items)

# Function to Generate association rules using those frequent itemset
rules = association_rules(frq_items, metric ="lift", min_threshold = 0.7)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules)

# pd.options.display.max_columns = None
# pd.options.display.max_rows = 220000

# pd.set_option('max_rows', 9999)
pd.set_option('max_colwidth', None)
pd.describe_option('max_colwidth')

print(rules.head(50))

# from mlxtend.frequent_patterns.fpgrowth import fpgrowth

# frequent_itemsets = fpgrowth(ds, min_support=0.01, use_colnames = True)
# frequent_itemsets.head(10)

# print(association_rules(frequent_itemsets, metric='lift', min_threshold=0.5))

# print(ds.head())

# ds_df = ds.copy()
# print(ds_df.pop("Raised By Both Parents"))
# print(ds_df.pop("Raised By Single Parent"))
# print(ds_df.head())

# frequent_itemsets1 = fpgrowth(ds_df, min_support=0.01, use_colnames = True)
# rules = association_rules(frequent_itemsets1, metric='lift', min_threshold=0.5)
# print(rules.head(30))