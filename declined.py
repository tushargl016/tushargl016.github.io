from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import preprocessing
import luigi
import urllib
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
import zipfile,io,os
import pandas as pd
import csv
import numpy as np
import luigi
import tarfile
import os,zipfile,io
import tarfile
import boto3
import botocore
import argparse
import sys
import datetime
from luigi.parameter import MissingParameterException

class downloadingdata(luigi.Task):
	def run(self):
		loginurl='https://www.lendingclub.com/account/login.action'

		url1="https://www.lendingclub.com/info/download-data.action"
		with requests.Session() as s:
			loginrequest=s.post(loginurl,data={'login_email':'heavenchild.88@gmail.com','login_password':'gaurika_123'})
			urlparse=s.get(url1)
			htmltext=urlparse.text

			#page = urllib.urlopen(urlparse).read()
			soup = BeautifulSoup(htmltext)
			soup.prettify()
			list_1=soup.find("div", {"id": "rejectedLoanStatsFileNamesJS"})
			if not os.path.exists('RejectLoanData_Part1'):
				os.makedirs('RejectLoanData_Part1')

			for a in list_1:
				print (a)

			list_2 = a.split('|')
			list_2.pop()


            # # print(list_1)

			url2="https://resources.lendingclub.com/"


			option_list = soup.find('select', id="rejectStatsDropdown")
			content=[str(x.text) for x in soup.find(id="rejectStatsDropdown").find_all('option')]
			print(content)


			count=0
			declined_load_data=pd.DataFrame()
			dat=pd.DataFrame()
			load_data_final=pd.DataFrame()
			for id in list_2:
					path = url2+id
					r =requests.get(path,stream=True)
					z = zipfile.ZipFile(io.BytesIO(r.content))
					x=z.extractall(os.path.join('RejectLoanData_Part1'))
					i = id[:-4]
					dat=pd.read_csv(os.path.join('RejectLoanData_Part1',i),skiprows=1,skipfooter=4)
					print(os.path.join('RejectLoanData_Part1',i))
					print(i)
					dat['timestamp']=content[count]
					count+=1
					if declined_load_data.empty:
						declined_load_data = dat
					else:
						declined_load_data=pd.concat([declined_load_data,dat],axis=0)

					path= ""
			declined_load_data.to_csv(self.output().path,index=False)

	def output(self):
		return luigi.LocalTarget('sceapeddeclineddata.csv')


class Cleaningdata(luigi.Task):
	def requires(self):
		yield downloadingdata()
	def run(self):
		#Cleaning data start

		declined_load_data=pd.read_csv(downloadingdata().output().path,encoding='ISO-8859-1')
		#renaming columns for ease of use
		declined_load_data=declined_load_data.rename(columns={'Zip Code':'zip_code','Loan Title':'loan_title','Application Date':'application_date','Employment Length':'emp_length','Policy Code':'policy_code','Amount Requested':'amount_requested','Debt-To-Income Ratio':'dti'})
		#finding the shape of the data and removing columns with more than 80% missing values
		sh=declined_load_data.shape
		for column in declined_load_data.columns:
			colmiss=(declined_load_data[column].isnull().sum())/sh[0]
			if colmiss>0.8:
				declined_load_data.drop(column,axis=1,inplace=True)

        #Removing zip code as it wont be useful in our analysis and it has xx in it so of no use
		declined_load_data.drop('zip_code',axis=1,inplace=True)
		#Filling missing risk scores with zero
		declined_load_data['Risk_Score'].fillna(0,inplace=True)
		#stripping % signs from dti to make it usefule for our algorithm and changing its data type to numeric
		declined_load_data['dti']=pd.Series(declined_load_data.dti ).str.replace('%', '').str.strip()

		#removing n/a values from emp_length and filling the missing values with 0
		#also removing signs like + < > and years and year to make it an int and usefule for our algorithm
		declined_load_data['emp_length'].replace('n/a',0,inplace=True)
		declined_load_data['emp_length']=pd.Series(declined_load_data.emp_length).str.replace('+', '').str.strip()
		declined_load_data['emp_length']=pd.Series(declined_load_data.emp_length).str.replace('<', '').str.strip()
		declined_load_data['emp_length']=pd.Series(declined_load_data.emp_length).str.replace('years', '').str.strip()
		declined_load_data['emp_length']=pd.Series(declined_load_data.emp_length).str.replace('year', '').str.strip()
		declined_load_data['emp_length'].fillna(0,inplace=True)
		declined_load_data['emp_length']=declined_load_data['emp_length'].astype(int)

		#Filling the missing categorical values with NA meaning not available
		declined_load_data['loan_title'].fillna('NA',inplace=True)
		declined_load_data['State'].fillna('NA',inplace=True)
		label_encoder = preprocessing.LabelEncoder()
		declined_load_data['State'] = label_encoder.fit_transform(declined_load_data['State'])
		#Filling the policy_code with maximum occuring value which is 0
		declined_load_data['policy_code'].fillna(0,inplace=True)
		#changing the datatype of dti after removing the % signs from it
		declined_load_data['dti']=declined_load_data['dti'].astype(float)

		#fetching the months and year values from the application date so that it can be used in our algorithm
		declined_load_data['application_date']=pd.to_datetime(declined_load_data['application_date'])
		def month_func(ts):
			return ts.month

		declined_load_data['app_month']=declined_load_data['application_date'].apply(month_func)
		def year_func(ts):
			return ts.year

		declined_load_data['app_year']=declined_load_data['application_date'].apply(year_func)

		#dropping the application date as all useful info has been extracted from it
		declined_load_data.drop('application_date',axis=1,inplace=True)
		#Removing the column loan_title as it contains 79000 categorical value and it makes no sense to one hot encode or label encode those values
		#For the same reason we dropped them in our approved loan data
		declined_load_data.drop('loan_title',axis=1,inplace=True)

		declined_load_data.to_csv(self.output().path,index=False)
	def output(self):
		return luigi.LocalTarget('CleanedDeclineddata.csv')
        #cleaning data end

class createzip(luigi.Task):
    def requires(self):
        yield Cleaningdata()
    def run(self):
        zipf=zipfile.ZipFile(self.output().path,'w',zipfile.ZIP_DEFLATED)
        zipf.write("CleanedDeclineddata.csv")
        zipf.close()
    def output(self):
        return luigi.LocalTarget('Declinedata.zip')


class uploadziptos3(luigi.Task):
	akey=luigi.Parameter()
	skey=luigi.Parameter()
	def requires(self):
		yield createzip()
	def run(self):
		if str(self.akey) == "1" or str(self.skey) == "1":
			print("please enter both access key and secret access key and rerun the program ")
			sys.exit()

		now=datetime.datetime.now()
		fin2=str(now).replace(":","")
		fin3=fin2.replace("-","")
		fin=fin3.replace(" ","")
		s3 = boto3.resource('s3')
		buckname="finalprojectdeclinelendingloandata"+str(fin)
		client = boto3.client('s3','us-west-2',aws_access_key_id=self.akey,aws_secret_access_key=self.skey)
		client.create_bucket(Bucket=buckname,CreateBucketConfiguration={'LocationConstraint':'us-west-2'})
		client.upload_file("Declinedata.zip", buckname, "Declinedata.zip")


if __name__ == '__main__':
    try:
        luigi.run()
    except MissingParameterException:
        print("Please provide Access Keys and Secret Access Keys")
        sys.exit()
