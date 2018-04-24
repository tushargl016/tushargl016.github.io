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


class getloandata(luigi.Task):
    def run(self):
        loginurl='https://www.lendingclub.com/account/login.action'
        url1="https://www.lendingclub.com/info/download-data.action"
        with requests.Session() as s:
            loginrequest=s.post(loginurl,data={'login_email':'heavenchild.88@gmail.com','login_password':'gaurika_123'})
            urlparse=s.get(url1)
            htmltext=urlparse.text

            #page = urllib.urlopen(urlparse).read()
            soup = BeautifulSoup(htmltext,"lxml")
            soup.prettify()
            list_1=soup.find("div", {"id": "loanStatsFileNamesJS"})

            if not os.path.exists('LoanData_Part1'):
                os.makedirs('LoanData_Part1')

            for a in list_1:
                continue

            list_2 = a.split('|')
            list_2.pop()


            # # print(list_1)

            url2="https://resources.lendingclub.com/"


            option_list = soup.find('select', id="loanStatsDropdown")
            content=[str(x.text) for x in soup.find(id="loanStatsDropdown").find_all('option')]
            #print(content)


            count=0
            load_data=pd.DataFrame()
            dat=pd.DataFrame()
            load_data_final=pd.DataFrame()
            print(list_2)
            for id in list_2:
                    path = url2+id
                    r =requests.get(path,stream=True)
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    x=z.extractall(os.path.join('LoanData_Part1'))
                    j=id.index(".zip")
                    m=id[:j]
                    n=m.index("/")
                    n+=1
                    i = id[n:j]
                    dat=pd.read_csv(os.path.join('LoanData_Part1',i),skiprows=1,skipfooter=4)
                    dat['timestamp']=content[count]
                    count+=1
                    if load_data.empty:
                        load_data = dat
                    else:
                         load_data=pd.concat([load_data,dat],axis=0)

                    path= ""
            load_data.to_csv(self.output().path,index=False)
    def output(self):
        return luigi.LocalTarget('Scrapedloandata.csv')




class missinganalysis(luigi.Task):
    def requires(self):
        yield getloandata()
    def run(self):
        load_data=pd.read_csv(getloandata().output().path,encoding='ISO-8859-1')
	    #starting data cleaning process
        #removing columns with more than 80% missing values
        sh=load_data.shape
        for column in load_data.columns:
            colmiss=(load_data[column].isnull().sum())/sh[0]
            if colmiss>0.8:
                load_data.drop(column,axis=1,inplace=True)

        #removing erroneous values like n/a by 0in emp_length
        #also removing signs like + < > and years and year to make it an int and usefule for our algorithm
        load_data['emp_length'].replace('n/a',0,inplace=True)
        load_data['emp_length']=pd.Series(load_data.emp_length).str.replace('+', '').str.strip()
        load_data['emp_length']=pd.Series(load_data.emp_length).str.replace('<', '').str.strip()
        load_data['emp_length']=pd.Series(load_data.emp_length).str.replace('years', '').str.strip()
        load_data['emp_length']=pd.Series(load_data.emp_length).str.replace('year', '').str.strip()
        load_data['emp_length'].fillna(0,inplace=True)
        load_data['emp_length']=load_data['emp_length'].astype(int)


        #zip_code consist of xx in it so stripping it to get the first 3 numeric values
        load_data.zip_code=pd.to_numeric(load_data.zip_code.str[:3])

        #Dropping title as it has too many categorical values and it makes no sense to label encode aor one hot encode these values
        load_data.drop('title',axis=1,inplace=True)

        #Replacing the last delinq with the max value for missing
        load_data['mths_since_last_delinq'].fillna(load_data['mths_since_last_delinq'].max(),inplace=True)

        #replacing the missing annual income by mean values
        load_data['annual_inc'].fillna(load_data['annual_inc'].mean(),inplace=True)

        #taking out the % from revol util so that we can make it a numeric value and filling out missing values with mean
        load_data['revol_util']=pd.Series(load_data.revol_util).str.replace('%', '').str.strip()
        load_data['revol_util'].fillna(load_data['revol_util'].median(),inplace=True)

        #Delleting the emp_title as it also had too many categorical values and it makes no sense to label encode aor one hot encode these values
        load_data.drop('emp_title',axis=1,inplace=True)

        #In the term column the first three values are numeric and rest contains xx so we are stripping that off
        load_data.term=pd.to_numeric(load_data.term.str[:3])

        #taking out the % from int rate so that we can make it a numeric value and filling out missing values with mean
        load_data['int_rate']=pd.Series(load_data.int_rate).str.replace('%', '').str.strip()

        #Filling the missing values with medians
        load_data['tot_coll_amt'].fillna(load_data['tot_coll_amt'].median(),inplace=True)
        load_data['tot_cur_bal'].fillna(load_data['tot_cur_bal'].median(),inplace=True)

        #filling the missing values with 0's
        load_data['earliest_cr_line'].fillna(0,inplace=True)
        load_data['last_pymnt_d'].fillna(0,inplace=True)

        #changing the type of values in home ownership column and converting them into two categories
        load_data['home_ownership'].replace({'OWN':1,'MORTGAGE':1,'RENT':0,'ANY':0,'OTHER':0,'NONE':0},inplace=True)

        #doing a similar change for verification_status
        load_data['verification_status'].replace({'Source Verified':1,'Verified':1,'Not Verified':0},inplace=True)

        #filling the missing values with medians
        load_data['total_rev_hi_lim'].fillna(load_data['total_rev_hi_lim'].median(),inplace=True)

        #taking out the year and months for issue date this will help us in our classification problem as well as summarization metrics
        load_data["issue_d"]=load_data['issue_d'].str.split("-")
        load_data['issue_year']=load_data['issue_d'].str[1]
        load_data['issue_month']=load_data['issue_d'].str[0]
        load_data['issue_month'].replace({'Jul':7,'Aug':8,'Mar':3,'Oct':10,'Apr':4,'Jun':6,'May':5,'Sep':9,'Jan':1,'Nov':11,'Feb':2,'Dec':12},inplace=True)

        #filling the missing values with 0's
        load_data['next_pymnt_d'].fillna(0,inplace=True)

        #dropping issue_d as important values have been taken from him
        load_data.drop('issue_d',axis=1,inplace=True)

        #We have a column called grade so column sub-grade has redundant values
        load_data.drop('sub_grade',axis=1,inplace=True)

        #we dont care about the initial status so dropping the columns
        load_data.drop('initial_list_status',axis=1,inplace=True)

        #creating the risk score column
        load_data['Risk_Score'] = (load_data.fico_range_low + load_data.fico_range_high) / 2

        #raplacing missing values with 0
        load_data['last_credit_pull_d'].fillna(0,inplace=True)

        #replacing rest values with 0's
        load_data.fillna(0,inplace=True)

        #We cannot use zip code as they are incomplete and from our past experience we saw that it gives bad values in our model
        load_data.drop('zip_code',axis=1,inplace=True)

        #We cannot use the id column for our prediction or anymodel so we drop it also their is no need for merging on key so we drop it
        load_data.drop('id',axis=1,inplace=True)

        #changing the datatypes of some features
        load_data[['issue_year','issue_month','loan_amnt','funded_amnt','funded_amnt_inv','annual_inc','delinq_2yrs','inq_last_6mths']]=load_data[['issue_year','issue_month','loan_amnt','funded_amnt','funded_amnt_inv','annual_inc','delinq_2yrs','inq_last_6mths']].astype('int64')
        load_data[['mths_since_last_delinq','open_acc','pub_rec','revol_bal','total_acc']]=load_data[['mths_since_last_delinq','open_acc','pub_rec','revol_bal','total_acc']].astype('int64')
        load_data[['int_rate','revol_util']]=load_data[['int_rate','revol_util']].astype(float)
        label_encoder = preprocessing.LabelEncoder()
        load_data['addr_state'] = label_encoder.fit_transform(load_data['addr_state'].astype('str'))
        load_data=load_data[load_data.int_rate!=0]
        load_data.to_csv(self.output().path,index=False,encoding="utf-8")
    def output(self):
        return luigi.LocalTarget('CleanedLoandata.csv')
        #cleaning data end

class FeatureEngineering(luigi.Task):
	def requires(self):
		yield missinganalysis()
	def run(self):
		model_features=pd.read_csv(missinganalysis().output().path,encoding='ISO-8859-1')
	#Begining feature selection

#We have more than 100 features but in our model we can only use the columns that a user can eneter
#From te dictionary we got an idea of the features that a user can enter and we build our model on that information and do feature selection on that




		Y = model_features.int_rate
		model_features.drop('int_rate', axis=1, inplace=True)
		user_entered_cols = ['loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc',
                                'verification_status', 'purpose', 'dti', 'delinq_2yrs',
                                'Risk_Score', 'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
                                'mths_since_last_major_derog', 'funded_amnt_inv', 'installment', 'application_type', 'pub_rec',
                                'addr_state']
		model_features = model_features[user_entered_cols]
		model_features=pd.get_dummies(model_features, columns=["purpose"])
		model_features=pd.get_dummies(model_features,columns=["application_type"])
		X = model_features._get_numeric_data()
		ranks={}
		def rank_to_dict(ranks, names, order=1):
			minmax = MinMaxScaler()
			ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
			ranks = map(lambda x: round(x, 2), ranks)
			return dict(zip(names, ranks ))

		lr = LinearRegression(normalize=True)
		lr.fit(X, Y)
		ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), user_entered_cols)

		ridge = Ridge(alpha=7)
		ridge.fit(X, Y)
		ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), user_entered_cols)

		lasso = Lasso(alpha=.05)
		lasso.fit(X, Y)
		ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), user_entered_cols)


		rlasso = RandomizedLasso(alpha=0.04)
		rlasso.fit(X, Y)
		ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), user_entered_cols)

        #stop the search when 5 features are left (they will get equal scores)
#		rfe = RFE(lr, n_features_to_select=5)
#		rfe.fit(X,Y)
#		ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), user_entered_cols, order=-1)

		rf = RandomForestRegressor()
		rf.fit(X,Y)
		ranks["RF"] = rank_to_dict(rf.feature_importances_, user_entered_cols)


		f, pval  = f_regression(X, Y, center=True)
		ranks["Corr."] = rank_to_dict(f, user_entered_cols)


		r = {}
		for name in user_entered_cols:
			r[name] = round(np.mean([ranks[method][name]
										for method in ranks.keys()]), 2)

		methods = sorted(ranks.keys())
		ranks["Mean"] = r
		methods.append("Mean")
		print ("\t%s" % "\t".join(methods))
		capoutput="\t".join(methods)
		f=open("feature.txt",'w')
		f.write(capoutput)
		f.write('\n')
		for name in user_entered_cols:
			capoutput=name + "\t" + " \t".join(map(str,
														[ranks[method][name] for method in methods]))
			f.write(capoutput)
			f.write("\n")
			print ("%s\t%s" % (name, "\t".join(map(str,
									[ranks[method][name] for method in methods]))))
		f.close()
		feature = pd.read_csv('feature.txt', sep='\t')
		feature.to_csv(self.output().path)
	def output(self):
		return luigi.LocalTarget('FeatureSelection.csv')

class createzip(luigi.Task):
    def requires(self):
        yield missinganalysis()
        yield FeatureEngineering()
    def run(self):
        zipf=zipfile.ZipFile(self.output().path,'w',zipfile.ZIP_DEFLATED)
        zipf.write("CleanedLoandata.csv")
        zipf.write('FeatureSelection.csv')
        zipf.close()
    def output(self):
        return luigi.LocalTarget('Loandata.zip')


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
		buckname="finalprojectlendingloandata"+str(fin)
		client = boto3.client('s3','us-west-2',aws_access_key_id=self.akey,aws_secret_access_key=self.skey)
		client.create_bucket(Bucket=buckname,CreateBucketConfiguration={'LocationConstraint':'us-west-2'})
		client.upload_file("Loandata.zip", buckname, "Loandata.zip")


if __name__ == '__main__':
    try:
        luigi.run()
    except MissingParameterException:
        print("Please provide Access Keys and Secret Access Keys")
        sys.exit()
