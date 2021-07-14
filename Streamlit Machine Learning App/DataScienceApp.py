import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection

st.set_option('deprecation.showPyplotGlobalUse', False)


from PIL import Image
import time


st.title('The Data Science Introductory App')
st.subheader('<---by Mladee--->')

#image = Image.open('logo.jpeg')
#st.image(image,use_column_width=False,width=150)


options = ['EDA','Visualization','Model Selection','App info and About me']
activity_choice = st.sidebar.selectbox('What would you like to perform: ', options)


def main():
	if activity_choice == 'EDA':
		st.subheader('Exploratory Data Analysis')
		get_file = st.file_uploader('Choose a file on your computer: ',type=['csv','xlsx','txt','html','json'])
		
		if get_file is not None:
			st.success("File Uploaded Succesfully!")
			data = pd.read_csv(get_file)
			st.dataframe(data.head(20))
		

	

		if st.checkbox('Show the shape of the dataset'):
			st.write(data.shape)

		if st.checkbox('Drop columns'):
			a = data.columns.to_list()
			how_many = st.slider('How many columns to drop: ',1,len(a))

			for i in range(how_many):
				cols = data.columns
				dropped_columns = st.selectbox('Choose columns to be dropped: ',cols)
				dropped = data.drop(dropped_columns, axis = 1)
				st.write(dropped)
				data = dropped


		
		if st.checkbox('Show all columns'):
			st.write(data.columns)
				
		if st.checkbox('Show summary of the data'):
			st.write(data.describe().T)

		if st.checkbox('Check for null values'):
			st.write(data.isnull().sum())

		if st.checkbox('Data Types'):
			st.write(data.dtypes)


		if st.checkbox('Check Correlation'):
			st.write(data.corr())

		

	elif activity_choice == 'Visualization':
		st.subheader('Visualization')
		get_file = st.file_uploader('Choose a file on your computer: ',type=['csv','xlsx','txt','html','json'])
		
		if get_file is not None:
			st.success("File Uploaded Succesfully!")
			data = pd.read_csv(get_file)
			st.dataframe(data.head(20))

		if st.checkbox('Drop columns'):
			a = data.columns.to_list()
			how_many = st.slider('How many columns to drop: ',1,len(a))

			for i in range(how_many):
				cols = data.columns
				dropped_columns = st.selectbox('Choose columns to be dropped: ',cols)
				dropped = data.drop(dropped_columns, axis = 1)
				st.write(dropped)
				data = dropped

		if st.checkbox('Display Heatmap(Ideal for less than 10 features)'):
			st.write(sns.heatmap(data.corr(), annot = True, square = True))
			st.pyplot()

		if st.checkbox('Display Pairplot(Ideal for less than 10 features)'):
			st.write(sns.pairplot(data, diag_kind = 'kde'))
			st.pyplot()

		if st.checkbox("Display Pie Chart(Ideal for target and categorical variables)"):
				all_columns=data.columns.to_list()
				pie_columns=st.selectbox("Select a column, NB: Select Target column",all_columns)
				pieChart=data[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pieChart)
				st.pyplot()

	elif activity_choice == 'Model Selection':
		st.subheader('Model Selection')
		get_file = st.file_uploader('Choose a file on your computer: ',type=['csv','xlsx','txt','html','json'])
		
		if get_file is not None:
			st.success("File Uploaded Succesfully!")
			data = pd.read_csv(get_file)
			st.dataframe(data.head(20))


		


		if st.checkbox('Replace Null Values'):
			replace_method = st.radio(
        "How do you wish to replace the null values?",
       ('ffill', 'bfill'))
			if replace_method == 'ffill':
				filled_ffill = data.fillna(method = 'ffill')
				st.write(filled_ffill)
				data = filled_ffill
			elif replace_method == 'bfill':
				filled_bfill = data.fillna(method = 'bfill')
				st.write(filled_bfill)
				data= filled_bfill
		if st.checkbox('Check for null values'):
			st.write(data.isnull().sum())

		st.write('Separate the dependent variable from the independent variables(X and y)')
		a = st.selectbox('Choose y column',data.columns)
		y = data[a]
		X = data.drop(a, axis = 1)
		st.write('Your y is: ',y)
		st.write('Your X is: ',X)
		

		bruh = ('SVM','KNN','Logistic Regression','Naive-Bayes')
		algorithms = st.sidebar.selectbox('Which algorithm would you like to use: ',(bruh))


		def check_algo(algo_name):
			params = dict()

			

			if algo_name == 'SVM':
				C = st.sidebar.slider('Value of C',0.01,100.1)
				params['C'] = C

			elif algo_name == 'KNN':
				K = st.sidebar.slider('Number of neighbors',3,50)
				params['K'] = K

			return params
			
		params = check_algo(algorithms)


		def algo_tuning(algo_name,params):
			model = None

		
			if algo_name == 'SVM':
				model = SVC(C = params['C'])
			elif algo_name == 'KNN':
				model = KNeighborsClassifier(n_neighbors=params['K'])
			elif algo_name == 'Logistic Regression':
				model = LogisticRegression()
			elif algo_name == 'Naive-Bayes':
				model = GaussianNB()
			return model

		classifier = algo_tuning(algorithms,params)

		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)

		classifier.fit(X_train,y_train)
		y_pred = classifier.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		st.write('Classifier name:',algorithms)
		st.write('Accuracy:', accuracy * 100, '%')


	elif activity_choice == 'App info and About me':

		st.success('''Thanks for taking your time to use my app.The app is pretty intuitive and it contains 3 parts:
			Exploratory Data Analysis, Data Visualization and Model Selection, where the user can apply machine learning models to a dataset.''')
		st.warning('Work in progress: PCA option, null values removal option, more models to be included,column selection tool, more visualization tool, more flexibility.')

		st.warning('''APP Info: To tune in the hyperparameters, simply adjust the slider and the accuracy will modify shortly.
			Do not upload a dataset that contains missing values, because the model will crash.''')

		st.error('About me: I am a passionate self-taught Data Science student that enjoys helping others and developing myself to become a great Machine Learning engineer ^.^')



			

















if __name__ =='__main__':
	main()



