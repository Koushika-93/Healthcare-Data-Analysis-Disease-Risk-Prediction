from pathlib import Path
import numpy as np, pandas as pd
import nbformat as nbf
root=Path(__file__).parent
rng=np.random.default_rng(42); n=600
age=rng.integers(18,81,n); sex=rng.choice(['Female','Male'],n); bmi=np.clip(rng.normal(27,5,n),16,48).round(1)
sbp=np.clip(rng.normal(124+(age-45)*.25,16,n),85,210).round().astype(int)
glucose=np.clip(rng.normal(105+(bmi-25)*1.5,22,n),65,240).round().astype(int)
chol=np.clip(rng.normal(195+(age-45)*.45,35,n),110,340).round().astype(int)
smoking=rng.choice(['No','Yes'],n,p=[.72,.28]); activity=rng.choice(['Low','Moderate','High'],n,p=[.35,.45,.2]); family=rng.choice(['No','Yes'],n,p=[.65,.35])
score=(age>55)*1.0+(bmi>=30)*1.0+(sbp>=140)*1.0+(glucose>=126)*1.3+(chol>=240)*.7+(smoking=='Yes')*.6+(activity=='Low')*.5+(family=='Yes')*.7+rng.normal(0,.7,n)
risk=np.where(score>=3.1,'High','Low')
df=pd.DataFrame({'Patient_ID':[f'P{i:04d}' for i in range(1,n+1)],'Age':age,'Sex':sex,'BMI':bmi,'Systolic_BP':sbp,'Glucose':glucose,'Cholesterol':chol,'Smoking':smoking,'Physical_Activity':activity,'Family_History':family,'Risk_Label':risk})
df.to_csv(root/'dataset/healthcare_risk_synthetic.csv',index=False)
cells=[]
cells.append(nbf.v4.new_markdown_cell('# Healthcare Data Analysis for Disease Risk Prediction\n**Educational demo:** synthetic patient data and rule-generated labels; not real diagnoses.'))
cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import classification_report, confusion_matrix, accuracy_score\ndf = pd.read_csv('../dataset/healthcare_risk_synthetic.csv')\ndf.head()"))
cells.append(nbf.v4.new_markdown_cell('## 1. Data inspection and cleaning'))
cells.append(nbf.v4.new_code_cell("print('Shape:', df.shape)\nprint(df.info())\nprint('Missing values:\\n', df.isnull().sum())\nprint('Duplicate rows:', df.duplicated().sum())\ndf = df.drop_duplicates().copy()\ndf.describe(include='all')"))
cells.append(nbf.v4.new_markdown_cell('## 2. Visualizations'))
cells.append(nbf.v4.new_code_cell("df['Risk_Label'].value_counts().plot(kind='bar', title='Synthetic Risk Label Counts')\nplt.xlabel('Label'); plt.ylabel('Records'); plt.tight_layout(); plt.show()"))
cells.append(nbf.v4.new_code_cell("sns.histplot(data=df, x='Age', hue='Risk_Label', bins=15, multiple='stack')\nplt.title('Age Distribution by Synthetic Label'); plt.tight_layout(); plt.show()"))
cells.append(nbf.v4.new_code_cell("numeric=['Age','BMI','Systolic_BP','Glucose','Cholesterol']\nsns.heatmap(df[numeric].corr(), annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Correlation Heatmap'); plt.tight_layout(); plt.show()"))
cells.append(nbf.v4.new_markdown_cell('## 3. Baseline machine-learning model\nThis predicts the synthetic label only. Do not use for diagnosis or treatment decisions.'))
cells.append(nbf.v4.new_code_cell("X=df.drop(columns=['Patient_ID','Risk_Label']); y=df['Risk_Label']\ncat_cols=X.select_dtypes(include='object').columns.tolist(); num_cols=X.select_dtypes(exclude='object').columns.tolist()\npreprocess=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num_cols),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat_cols)])\nmodel=Pipeline([('prep',preprocess),('clf',RandomForestClassifier(n_estimators=150,random_state=42,class_weight='balanced'))])\nX_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)\nmodel.fit(X_train,y_train); pred=model.predict(X_test)\nprint('Accuracy:',round(accuracy_score(y_test,pred),3))\nprint(classification_report(y_test,pred,zero_division=0))\nprint('Confusion matrix:\\n',confusion_matrix(y_test,pred,labels=['Low','High']))"))
cells.append(nbf.v4.new_markdown_cell('## Conclusion\nThis workflow demonstrates data loading, cleaning checks, visualization, and model evaluation. Synthetic labels cannot establish real-world clinical accuracy.'))
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbf.write(nb,root/'notebook/healthcare_risk_analysis.ipynb')
