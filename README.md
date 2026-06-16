df.info() - Understanding the description of the dataset
Attribute		Description & Data Type     Action
index       index for entry - Nominal       Remove index
student_id		Unique ID for each student - Nominal    Remove id
number_of_siblings	Number of siblings - Ratio      Keep/Remove
direct_admission	Mode of entering the school - Nominal       Keep boolean encoding/Remove
CCA			Enrolled CCA - Nominal      Keep categorical encoding/Remove
learning_style		Primary learning style - Nominal        Keep categorical encoding/Remove
tuition			Indication of whether the student has a tuition - Nominal       Keep boolean encoding/Remove
final_test		Student's O-level mathematics examination score - Ratio     Predicted Value
n_male			Number of male classmates - Ratio       Keep/Remove
n_female		Number of female classmates - Ratio     Keep/Remove
gender			Gender type - Nominal       Keep boolean encoding/Remove
age			Age of the student - Ratio      Keep/Remove
hours_per_week		Number of hours student studies per week - Ratio    Keep/Remove
attendance_rate		Attendance rate of the student (%) - Ratio      Keep/Remove
sleep_time		Daily sleeping time (hour:minutes) - Interval(?)    Feature Engineering to sleep_hours_daily
wake_time		Daily waking up time (hour:minutes) - Interval(?)   Feature Engineering to sleep_hours_daily
mode_of_transport	Mode of transport to school - Nominal   Keep categorical encoding/Remove
bag_color		Colours of student's bag - Nominal      Remove


df.duplicated().sum()
No duplicates after checking - no need to remove duplicates

df.isnull().sum()
CCA                   3829
final_test             495
attendance_rate        778
1. Remove all null final_test since that is the predicted value, without the final_test score the entry is not useful. It could have been that the student did not take the test hence no final_test score, giving a synthetic score like mean or median will influence the model drastically, hence the decision to remove the entries.
2. CCA replace the null with 'NONE' for students with 'NaN' CCA. We can make the assumption that a null CCA entry would suggest that the student is not part of any CCA, hence we will make that another category.
3. attendance_rate null values are not explainable via the dataset. Propose to use the mean or median value to fill the null attendance_rate entries

df.describe().T
feature     count	mean	std	min	25%	50%	75%	max
age	15900.0	15.213459	1.758941	-5.0	15.00	15.0	16.00	16.0
1. min age was observed to be -5, which is not a possible age value. Total of 451 entries with age less than 15. Transform the 5 and -5 ages to 15, then 6 and -6 to 16, under the assumption that the age was collected mistakenly. Since most of the erroneous data were 6,5,4,-5,-6 and the expected age is between the range of 14 and 16, we can make a reasonable assumption to clean and transform the data accordingly to match the reasonable age instead.

Univariate Feature Analysis
direct_admission - Yes & No
CCA - Sports, Arts, Club, ARTS, SPORTS, CLUB, NONE
learning_style - Visual & Auditory
gender - Male & Female
tuition - Yes, No, Y, N
mode_of_transport - private transport, public transport, walk
bag_color - yellow, green, white, red, blue, black 

#Bivariate Feature Analysis
All features against final_test (scatter plot & box plot)

#Correlation Analysis


#Data Cleaning 
CCA - Sports, Arts, Club, ARTS, SPORTS, CLUB, NONE
1. Transform to use standardized string, Pascal Case formatting - Sports, Arts, Club, None
2. Consideration - CCA vs No CCA instead of categorical encoding 
tuition - Yes, No, Y, N
1. Transform to use standardized string, Pascal Case formatting - Yes & No
gender - Male & Female 
1. Remove due to poor relationship against final_test
mode_of_transport - private transport, public transport, walk
1. Remove due to poor relationship against final_test
bag_color - yellow, green, white, red, blue, black 
1. Remove due to poor relationship against final_test

age
1. Remove due to poor relationship against final_test 
n_male & n_female
1. Highly correlated - Remove due to poor relationship against final_test 

#Feature Engineering 
sleep_time & wake_time
1. Transform to sleep_hours_daily

Data Splitting
Feature Scaling 
Feature Encoding 