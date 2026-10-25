import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title='Healthcare Risk Analysis', page_icon='🩺', layout='wide')
st.title('Healthcare Data Analysis for Disease Risk Prediction')
st.warning('Educational demo only: this app uses synthetic data and a rule-generated label. It is not a medical device and cannot diagnose disease or estimate your personal clinical risk.')
DATA = Path(__file__).resolve().parents[1] / 'dataset' / 'healthcare_risk_synthetic.csv'
df = pd.read_csv(DATA)
st.caption('Dataset: 600 synthetic patient records created for this project.')
col1,col2,col3 = st.columns(3)
col1.metric('Patient records', f'{len(df):,}')
col2.metric('High synthetic-risk labels', f'{(df.Risk_Label == "High").sum():,}')
col3.metric('Average age', f'{df.Age.mean():.1f} years')
st.subheader('Explore the dataset')
choice = st.selectbox('Filter by synthetic label', ['All','Low','High'])
view = df if choice == 'All' else df[df.Risk_Label == choice]
st.dataframe(view, use_container_width=True, hide_index=True)
a,b = st.columns(2)
with a:
    st.subheader('Synthetic label distribution')
    counts = df['Risk_Label'].value_counts()
    fig,ax=plt.subplots(); counts.plot(kind='bar',ax=ax); ax.set_xlabel('Label'); ax.set_ylabel('Records'); st.pyplot(fig)
with b:
    st.subheader('Average health measures by label')
    measures = df.groupby('Risk_Label')[['BMI','Systolic_BP','Glucose','Cholesterol']].mean().round(1)
    st.dataframe(measures, use_container_width=True)
st.subheader('Age distribution')
fig,ax=plt.subplots(); df.boxplot(column='Age', by='Risk_Label', ax=ax); ax.set_title('Age by synthetic label'); ax.set_xlabel('Label'); ax.set_ylabel('Age'); fig.suptitle(''); st.pyplot(fig)
st.download_button('Download synthetic dataset (CSV)', df.to_csv(index=False).encode('utf-8'), 'healthcare_risk_synthetic.csv', 'text/csv')
st.info('For real clinical research, use ethically approved, de-identified data and validate models with qualified healthcare professionals.')
