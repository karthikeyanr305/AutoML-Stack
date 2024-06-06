import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import seaborn as sb
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
import time

import json

#import models:
from clean_auto import clean_auto
from model_visualize import visualize_model
from model_files.logisticRegression import logisticRegression
from model_files.decisionTree import decisionTree
from model_files.randomForestClassifier import randomForestClassifier
from model_files.adaBoost import adaBoost
from model_files.XGBoost import XGBoost
from model_files.naiveBayes import naiveBayes
import joblib

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

print("ENtering applicaton")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout="wide")


def call_cust(custom_list, file):
    loaded_model = joblib.load(file)
    #custom_list1 = [int(float((i))) for i in custom_list]
    custom_list1 = np.array(custom_list)
    test1 = custom_list1.reshape(1,-1)
    result = loaded_model.predict(test1)
    result1 = loaded_model.predict_proba(test1)
    
    
    if int(result) == 0:
        output = "Readmission not required within 30 days!"
    else:
        output = "Readamission required within 30 days!"
    return output, result1.astype(float)


@st.cache_data(allow_output_mutation=True, show_spinner=False)
def plot_treemap(fraud_counts):



    fig_treemap = px.treemap(fraud_counts, path=[px.Constant('All Categories'), 'category'], values='Fraud count',
                 title='Treemap of Fraud Counts by Category',
                 color='Fraud count',
                 color_continuous_scale='Blues')
    fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))

    st.plotly_chart(fig_treemap, use_container_width=True)

#@st.cache_data
def plot_pie(dataset):
    labels = ["Authentic Transaction", "Fraudulent Transaction"]
    values = dataset["is_fraud"].value_counts()
    colors = ['mediumturquoise', 'lightgreen']
    # pull is given as a fraction of the pie radius
    
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.2], domain={'x': [0.05, 0.95], 'y': [0.01, 0.95]})])

    fig_pie.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=12,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    
    # Update layout with title adjustments
    fig_pie.update_layout(
        title={
            'text': 'Authentic vs Fraudulent Transactions',
            'y':0.97,  # moves the title upwards
            'x':0.5,  # centers the title
            'xanchor': 'center',
            'yanchor': 'top'
        }

    )
    #fig.show()
    
    # Plot!
    time.sleep(0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

#@st.cache_data
def plot_combined(fraud_stats):
    # Create subplots
    fig_comb = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar chart for the count of fraud transactions
    fig_comb.add_trace(
        go.Bar(x=fraud_stats['year_month'], y=fraud_stats['fraud_count'], name='Fraud Count'),
        secondary_y=False,
    )

    # Add area chart for the sum of fraud amounts
    fig_comb.add_trace(
        go.Scatter(x=fraud_stats['year_month'], y=fraud_stats['fraud_amount'], name='Fraud Amount', fill='tozeroy'),
        secondary_y=True,
    )

    # Add figure titles and axis labels
    fig_comb.update_layout(
        title_text='Fraud Transactions Analysis Over Time',
        xaxis_title='Year-Month',
        legend_title_text='Metrics'
    )

    # Set y-axes titles
    fig_comb.update_yaxes(title_text='Count of Fraud Transactions', secondary_y=False)
    fig_comb.update_yaxes(title_text='Sum of Fraud Amounts', secondary_y=True)

    # Show the figure
    st.plotly_chart(fig_comb, use_container_width=True)

#@st.cache_data
def plot_geo_map(fraud_counts):

    # Create the scatter geo plot
    fig_geo = go.Figure(data=go.Scattergeo(
        lon = fraud_counts['long'],
        lat = fraud_counts['lat'],
        text = fraud_counts['city'] + ', ' + fraud_counts['state'] + ': ' + fraud_counts['fraud_count'].astype(str) + ' frauds',
        mode = 'markers',
        marker = dict(
            size = fraud_counts['fraud_count'],
            sizemode = 'area',
            sizeref = 2.*max(fraud_counts['fraud_count'])/(40.**2),
            sizemin = 4,
            color = fraud_counts['fraud_count'],
            colorscale = 'Reds',
            line_width = 1,
            line_color='black',
            showscale=True
        )
    ))

    # Update layout for a clearer view
    fig_geo.update_layout(
        title = 'Number of Fraudulent Transactions by City in the US',
        geo_scope='usa',  # limit map scope to USA
    )

    # Show the figure
    st.plotly_chart(fig_geo, use_container_width=True)

#@st.cache_data
def plot_geo_map2(fraud_stats):

    # Create the scatter geo plot
    fig_geo2 = go.Figure(data=go.Scattergeo(
        lon = fraud_stats['long'],
        lat = fraud_stats['lat'],
        text = fraud_stats['city'] + ', ' + fraud_stats['state'] +
            ': ' + fraud_stats['Fraud_count'].astype(str) + ' frauds, $' +
            fraud_stats['Fraud_sum'].astype(str),
        mode = 'markers',
        marker = dict(
            size = fraud_stats['Fraud_sum'],
            sizemode = 'area',
            sizeref = 2.*max(fraud_stats['Fraud_sum'])/(40.**2),
            sizemin = 4,
            color = fraud_stats['Fraud_count'],
            colorscale = 'Reds',
            line_width = 1,
            line_color='black',
            showscale=True
        )
    ))

    # Update layout for a clearer view
    fig_geo2.update_layout(
        title = 'Fraudulent Transactions by City in the US',
        geo_scope='usa',  # limit map scope to USA
    )

    # Show the figure
    #fig_geo2.show()
    st.plotly_chart(fig_geo2, use_container_width=True)

#@st.cache_data
def stream_clean():
    clean_text0 = "Cleaning...\n"
    clean_text1 = "1.Formatting Data types.\n"
    clean_text2 = "2.Dropping unnecessary columns.\n"
    clean_text3 = "3.Formatting missing values.\n"
    clean_text4 = "4.Encoding cateogrical features."
    #clean_text = clean_text0 + clean_text1 + clean_text2 + clean_text3 + clean_text4 
    clean_text = [clean_text0, clean_text1, clean_text2, clean_text3, clean_text4]
    
    for clean in clean_text:
        for word in clean.split(" "):
            yield word + " "
            time.sleep(0.1)
        yield '\n'

# Creating function for streaming imblance caution
#@st.cache_data
def stream_imbalance_caution():
    imbalance_caution = "There seems to be a class imbalance. Would you like to improve the model performance by using SMOTE?\n"
    imbalance = [imbalance_caution]       
    for clean in imbalance:
        for word in clean.split(" "):
            yield word + " "
            time.sleep(0.1)
        yield '\n'

@st.cache_data(experimental_allow_widgets=True)
def stream_imbalance_caution_static():
    st.write("There seems to be a class imbalance. Would you like to improve the model performance by using SMOTE?\n")

#@st.cache_data
def compare_model_smote(model_name, isSMOTE, file_size):

    if True:
        # Load JSON data from files
        with open('model_metrics/{}_metrics_{}_False.json'.format(model_name, file_size), 'r') as file:
            data1 = json.load(file)

        with open('model_metrics/{}_metrics_{}_True.json'.format(model_name, file_size), 'r') as file:
            data2 = json.load(file)

        # Extracting metric names and values
        metric_names = data1['metric_names']
        values1 = data1['metric_values']
        values2 = data2['metric_values']

        # Creating a grouped bar chart
        fig = go.Figure()

        # Adding the first set of bars for the first file
        fig.add_trace(go.Bar(
            x=metric_names,
            y=values1,
            name='SMOTE metrics',
            marker_color='navy'
        ))

        # Adding the second set of bars for the second file
        fig.add_trace(go.Bar(
            x=metric_names,
            y=values2,
            name='non-SMOTE metrics',
            marker_color='lightblue'
        ))

        # Updating the layout for clear visualization
        fig.update_layout(
            title='Comparison of Metrics with and without SMOTE',
            xaxis_title='Metric',
            yaxis_title='Value',
            barmode='group',
            xaxis={'categoryorder':'total descending'}
        )

        # Show the figure
        #fig.show()
        st.plotly_chart(fig, use_container_width=True)

#@st.cache_data
def plot_compare_all(isSMOTE, file_size):
    # Function to load data from a JSON file
    def load_data(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data

    # Load data from all four files
    data1 = load_data('model_metrics/{}_metrics_{}_{}.json'.format('LR', file_size, isSMOTE))
    data2 = load_data('model_metrics/{}_metrics_{}_{}.json'.format('RF', file_size, isSMOTE))
    data3 = load_data('model_metrics/{}_metrics_{}_{}.json'.format('XG', file_size, isSMOTE))

    model_list = ["Logistic Regression", "Random Forest", "XGBoost"]


    # Create a DataFrame for plotting
    df = pd.DataFrame()
    for i, data in enumerate([data1, data2, data3], start=0):
        temp_df = pd.DataFrame({
            'Metric': data['metric_names'],
            'Value': data['metric_values'],
            'Model': model_list[i]
        })
        df = pd.concat([df, temp_df], ignore_index=True)

    # Replace None or NaN values with 0 in the Value column
    df['Value'].fillna(0, inplace=True)

    # Plotting with Plotly Express
    fig = px.bar(df, x='Metric', y='Value', color='Model', barmode='group',
                category_orders={"Model": ["Logistic Regression", "Random Forest", "XGBoost"]},
                color_discrete_sequence=["lightblue", "dodgerblue", "deepskyblue"])


    # Adjust bar width to reduce overlap
    #fig.update_traces(width=0.15)  # Adjust bar width here
    

    # Adjusting bar groups closer to each other and update layout for better readability
    fig.update_layout(
        title='Comparison of All Model Performance',
        xaxis_title='Metric',
        yaxis_title='Value',
        bargap=0.15,  # distance between bars of adjacent location coordinates
        bargroupgap=0.1  # distance between bars of the same location coordinate
    )

    # Show the plot
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)

def basicEDA2(dataset0, file_size):

    # converting to datetime format
    dataset0["trans_date_trans_time"] = pd.to_datetime(dataset0["trans_date_trans_time"])
    dataset0["dob"] = pd.to_datetime(dataset0["dob"])

    #Dropping unnecessary columns
    #dataset.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
    reqd_cols = ['category', 'amt', 'gender', 'lat', 'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']
    

    dataset = dataset0.copy(deep=True)

    time.sleep(1)
    
    dataset = dataset[reqd_cols]

    #Drop all rows that contain missing values 
    dataset = dataset.dropna(ignore_index=True)

    #Encoding cateogrical variables
    encoder = LabelEncoder()
    #dataset["merchant"] = encoder.fit_transform(dataset["merchant"])
    dataset["category"] = encoder.fit_transform(dataset["category"])
    dataset["gender"] = encoder.fit_transform(dataset["gender"])
    dataset["job"] = encoder.fit_transform(dataset["job"])

    st.write_stream(stream_clean)

    view_clean  = st.checkbox("View Sample Dataset After Cleaning")

    if view_clean:
        st.header('Sample Data - Cleaned')
        st.write(dataset.head())


    # Filter and group by 'category' to count 'isFraud = 1' instances
    fraud_counts_tree = dataset0[dataset0['is_fraud'] == 1].groupby('category').size().reset_index(name='Fraud count')

    fraud_counts_and_amounts = dataset0[dataset0['is_fraud'] == 1].groupby('category').agg(
    Fraud_count=('category', 'size'),  # Count of fraudulent transactions per category
    Fraud_sum=('amt', 'sum')       # Sum of amount for fraudulent transactions per category
    ).reset_index()

    fraud_df = dataset0[dataset0['is_fraud']== 1].copy()

    fraud_df['year_month'] = fraud_df['trans_date_trans_time'].dt.to_period('M')

    # Group by year_month to calculate counts and sums
    fraud_stats = fraud_df.groupby(['year_month']).agg({
        'is_fraud': 'count',  # Count of fraud transactions
        'amt': 'sum'      # Sum of amounts for fraud transactions
    }).rename(columns={'is_fraud': 'fraud_count', 'amt': 'fraud_amount'})

    
    # Reset index to make 'year_month' a column
    fraud_stats.reset_index(inplace=True)
    fraud_stats['year_month'] = fraud_stats['year_month'].astype(str)  # Convert to string for plotting

    # Find the index of the maximum amount of fraud transactions
    month_max_amt = fraud_stats['fraud_amount'].idxmax()
    month_max_cnt = fraud_stats['fraud_count'].idxmax()

    month_max_amt2 = fraud_stats[['year_month', 'fraud_amount']].sort_values(by='fraud_amount', ascending=False).head(3)
    month_max_cnt2 = fraud_stats.sort_values(by=['fraud_count', 'fraud_amount'], ascending=[False, False])[['year_month', 'fraud_count']].head(3)

    # Retrieve the month for count and amount using the index
    month_amt = fraud_stats.loc[month_max_amt, 'year_month']
    month_cnt = fraud_stats.loc[month_max_cnt, 'year_month']

    # Group by city and state, and count fraud occurrences
    #fraud_counts_loc = fraud_df.groupby(['city', 'state']).size().reset_index(name='fraud_count')   
    fraud_counts_loc = fraud_df.groupby(['city', 'state', 'lat', 'long']).size().reset_index(name='fraud_count')

    # Group by city to calculate sum of amounts and count of frauds
    fraud_stats_loc = fraud_df.groupby(['city', 'state', 'lat', 'long']).agg({
        'amt': 'sum',  # Sum of amounts for fraud transactions
        'is_fraud': 'count'  # Count of fraud transactions
    }).rename(columns={'is_fraud': 'Fraud_count', 'amt': 'Fraud_sum'}).reset_index()

    # Find the index of the maximum amount of fraud transactions
    idx_max_amt = fraud_stats_loc['Fraud_sum'].idxmax()

    # Retrieve the city name and amount using the index
    max_fraud_city = fraud_stats_loc.loc[idx_max_amt, 'city']
    max_fraud_amt = fraud_stats_loc.loc[idx_max_amt, 'Fraud_sum']

    top_three_fraud_cities_cnt = fraud_stats_loc[['city', 'Fraud_sum']].sort_values(by='Fraud_sum', ascending=False).head(3)
    top_three_fraud_cities_sum = fraud_stats_loc[['city', 'Fraud_count']].sort_values(by='Fraud_count', ascending=False).head(3)

    # Top 3 Categories by fraud
    top_three_fraud_categories = fraud_counts_tree[['category', 'Fraud count']].sort_values(by='Fraud count', ascending=False).head(3)
    top_three_fraud_categories_cnt = fraud_counts_and_amounts[['category', 'Fraud_count']].sort_values(by='Fraud_count', ascending=False).head(3)
    top_three_fraud_categories_sum = fraud_counts_and_amounts[['category', 'Fraud_sum']].sort_values(by='Fraud_sum', ascending=False).head(3)
    


    explore_data  = st.toggle("Perform EDA")

    if explore_data:
        st.write("Voila!")

        top_left, top_right = st.columns((2,1.5))

        with top_left:
            plot_treemap(fraud_counts_tree)
        
        with top_right:
            plot_pie(dataset)

        plot_combined(fraud_stats)

        #plot_geo_map(fraud_counts_loc)

        plot_geo_map2(fraud_stats_loc)


    
    
        st.header("Important Findings")
        st.subheader('City with highest amount of Fraudulent transactions')
        st.write("\n")
        #st.markdown("{} - ${}".format(max_fraud_city, max_fraud_amt))
        
        city_fraud_left, city_fraud_right, _1 = st.columns((0.4,0.4, 0.2), gap="small")
        with city_fraud_left:
            st.dataframe(top_three_fraud_cities_cnt.set_index('city'))
        with city_fraud_right:
            st.dataframe(top_three_fraud_cities_sum.set_index('city'))

        
        st.subheader('Top 3 categories by number of fraudulent transactions')
        st.write("\n")
        #st.write(top_three_fraud_categories, index=False)
        #st.dataframe(top_three_fraud_categories.set_index('category'))
        cat_fraud_left, cat_fraud_right, _2 = st.columns((0.4,0.4, 0.2), gap="small")
        with cat_fraud_left:
            st.dataframe(top_three_fraud_categories_cnt.set_index('category'))
        with cat_fraud_right:
            st.dataframe(top_three_fraud_categories_sum.set_index('category'))
        


        st.subheader('When did the highest number of fraudlent transctions happen?')
        st.write("\n")
        month_fraud_left, month_fraud_right, _2 = st.columns((0.4,0.4, 0.2), gap="small")
        with month_fraud_left:
            st.dataframe(month_max_amt2.set_index('year_month'))
        with month_fraud_right:
            st.dataframe(month_max_cnt2.set_index('year_month'))

        '''st.markdown("{} : {} transactions".format(month_cnt, fraud_stats['fraud_count'].max()))
        st.markdown("{} : ${}".format(month_amt, fraud_stats['fraud_amount'].max().round(2)))'''

    st.write("\n")

    #st.write_stream(stream_imbalance_caution)

    stream_imbalance_caution_static()



    smote_on  = st.checkbox("Use SMOTE", help="SMOTE (Synthetic Minority Over-sampling Technique) is a method that synthetically generates samples from the minority class in imbalanced datasets to improve model learning and reduce bias towards the majority class. This is especially useful in scenarios like fraud detection, where rare events are critical to identify accurately.")
    st.session_state.isSMOTE = True
    #isSMOTE = True
    if smote_on:
        st.write("Good Choice!")
        print('isSMOTE: ', st.session_state.isSMOTE)
    else:
        st.session_state.isSMOTE = False

    st.session_state.X = dataset.iloc[:, :-1]
    st.session_state.y = dataset.iloc[:, -1]

    '''model_option = st.radio("Select the model to train your dataset with:", ("Logistic Regression", "Decision Tree", "Random Forest Classifier", "Ada Boost",
                    "XGBoost", "Naive Bayes"), help = "Logistic Regression: Linear model for classification and regression." +
                          "\n\n Decision Tree: Tree-based model that makes decisions based on feature values." + 
                          "\n\n Random Forest Classifier: Ensemble of decision trees for classification." +
                            "\n\n Ada Boost: Ensemble model that combines weak learners to create a strong learner." + 
                            "\n\n XGBoost: Optimized gradient boosting framework for improved model performance." + 
                            "\n\n Naive Bayes: Probabilistic model based on Bayes theorem for classification.")'''
    
    st.session_state.model_option = None
    st.session_state.train_model = None 

    st.session_state.model_option = st.radio("Select the model to train your dataset with:", ("Logistic Regression", "Random Forest Classifier",
                    "XGBoost"), help = "Logistic Regression: Linear model for classification and regression." + 
                          "\n\n Random Forest Classifier: Ensemble of decision trees for classification." +
                            "\n\n XGBoost: Optimized gradient boosting framework for improved model performance.")
    
    model_metrics = {}

    st.session_state.train_model = st.toggle("Train Model!")#st.button("Train Model!")

    if st.session_state.train_model:


        if st.session_state.model_option == "Logistic Regression":
            
            st.write(f"<p style='color:#0FF900'><strong>Training with Logistic Regression!</strong></p>", unsafe_allow_html=True)
            
            model, X, y, X_train, X_test, y_train, y_test = logisticRegression(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)
            #joblib.dump(model, 'models/lr.joblib')
            lr_metrics = visualize_model(model, X, y, X_train, X_test, y_train, y_test, 'LR', st.session_state.isSMOTE, file_size)

            
            compare_model_smote('LR', st.session_state.isSMOTE, file_size)
            


        elif st.session_state.model_option == "Decision Tree":

            st.write(f"<p style='color:#0FF900'><strong>Training with Decision Tree!</strong></p>", unsafe_allow_html=True)
            
            decisionTree(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)
        
        elif st.session_state.model_option == "Random Forest Classifier":
            
            st.write(f"<p style='color:#0FF900'><strong>Training with Random Forest Classifier!</strong></p>", unsafe_allow_html=True)
        
            model, X, y, X_train, X_test, y_train, y_test = randomForestClassifier(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)
            rf_metrics = visualize_model(model, X, y, X_train, X_test, y_train, y_test, 'RF', st.session_state.isSMOTE, file_size)

            compare_model_smote('RF', st.session_state.isSMOTE, file_size)


        elif st.session_state.model_option == "Ada Boost":
            
            st.write(f"<p style='color:#0FF900'><strong>Training with AdaBoost!</strong></p>", unsafe_allow_html=True)
            
            adaBoost(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)

        elif st.session_state.model_option == "XGBoost":

            st.write(f"<p style='color:#0FF900'><strong>Training with XGBoost!</strong></p>", unsafe_allow_html=True)
            
            model, X, y, X_train, X_test, y_train, y_test = XGBoost(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)
            xg_metrics = visualize_model(model, X, y, X_train, X_test, y_train, y_test, 'XG', st.session_state.isSMOTE, file_size)

            compare_model_smote('XG', st.session_state.isSMOTE, file_size)

        
        elif st.session_state.model_option == "Naive Bayes":
            
            st.write(f"<p style='color:#0FF900'><strong>Training with Naive Bayes!</strong></p>", unsafe_allow_html=True)
            
            naiveBayes(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)

    #compare_models  = st.toggle("Compare model performance with and without SMOTE")
    compare_all_models  = st.toggle("Compare model performance with other models")

    if compare_all_models:
        plot_compare_all(st.session_state.isSMOTE, file_size)
        
    wrap_up = st.button("Wrap up!", type="primary")
    block_chain_url = "https://www.fraud.com/post/real-time-fraud-prevention"
    wrap_content = """Diving into credit card fraud detection revealed some eye-openers. Using SMOTE really switched things up, making our models smarter at spotting frauds in an ocean of transactions. Without it, accuracy looked great, but it was like only seeing the tip of the iceberg—lots left unseen below the surface.

XGBoost stole the show, proving once again why it's a go-to for data scientists. And guess what? The dollar amount of transactions was a game-changer for model accuracy.

What's next? Imagine using AI that learns on the fly, catching fraudsters as they evolve. Integrating blockchain technology to trace transaction pathways in real-time, enhancing the predictive power of our AI systems. The future's about making these smart tools even smarter, ensuring our digital transactions are safe while tech plays a crucial role behind the scenes of our everyday online interactions.

Feel free to deep dive into how blockchain is currently being used in real-time fraud prevention. [Link](%s) """

    if wrap_up:
        st.header("Wrapping Up Our Fraud Detection Journey")
        st.markdown(wrap_content %block_chain_url)
        
@st.cache_data
def read_data(data_option):
    
    dataset_read = None
    if data_option == "Default Dataset - 10k":
        dataset_read = pd.read_csv('./data/credit_card_transactions/fraud_mini.csv')
    
    elif data_option == "Default Dataset - 100k":
        dataset_read = pd.read_csv('./data/credit_card_transactions/fraud_mini100k.csv')

    return dataset_read

def click_start():
    if st.session_state.lets_start: 
        st.session_state.lets_start = False
    else:
        st.session_state.lets_start = True

def app():

    if 'lets_start' not in st.session_state:
        st.session_state.lets_start = False

    st.button("Let's Start!", type='primary', on_click=click_start)

    #st.button('Click me', on_click=click_start)

    fraud_url = "https://www.bankrate.com/credit-cards/news/credit-card-fraud-statistics/#protect"

    if st.session_state.lets_start:
        intro_content = """Credit card fraud is a growing issue, draining billions each year from the global economy and affecting countless victims.
        As digital transactions increase, so does the complexity and frequency of fraud, challenging outdated detection methods and demanding more sophisticated solutions.
            This project's insights and solutions are particularly useful for financial institutions, security experts, and technology developers seeking to enhance their fraud
            detection systems. [Link](%s)"""
        problem_content = """The challenge is significant: not only do fraudsters continuously refine their tactics, but traditional detection systems also struggle to keep pace.
        The result? Financial losses and eroded trust. We utilize advanced analytics and machine learning, including techniques like SMOTE to tackle imbalanced data for
            predictive accuracy, to enhance detection capabilities. Our goal is to not just detect fraud, but to predict and prevent it, ensuring security and trust in every transaction."""
        st.markdown(intro_content % fraud_url)
        st.header('Problem Statement')
        st.markdown(problem_content)

        data_option = st.selectbox("Select an option to make prediction", ["--Select--", "Default Dataset - 10k", "Default Dataset - 100k", "Upload Dataset - CSV"])

        dataset = read_data(data_option)
        
        if data_option == "Default Dataset - 10k":
            #dataset_train = pd.read_csv('./data/credit_card_transactions/fraudTrain.csv')
            #dataset_test = pd.read_csv('./data/credit_card_transactions/fraudTest.csv')
            #dataset = pd.concat([dataset_train, dataset_test], axis=0, ignore_index=True)

            dataset = pd.read_csv('./data/credit_card_transactions/fraud_mini.csv')
            #dataset = pd.read_csv('./data/credit_card_transactions/fraudTest_mini.csv')

            st.header('About the dataset')

            about_data = """This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020.
            It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants. This subset of the dataset was created by randomly sampling from the original data."""
            st.markdown(about_data)

            st.link_button("Go to Data", "https://www.kaggle.com/datasets/kartik2112/fraud-detection/data")

            st.header('Sample Raw Data')
            st.write(dataset.head(10))
            #st.write(dataset.columns)
            
            basicEDA2(dataset, '10k')

        elif data_option == "Default Dataset - 100k":
            #dataset_train = pd.read_csv('./data/credit_card_transactions/fraudTrain.csv')
            #dataset_test = pd.read_csv('./data/credit_card_transactions/fraudTest.csv')
            #dataset = pd.concat([dataset_train, dataset_test], axis=0, ignore_index=True)

            dataset = pd.read_csv('./data/credit_card_transactions/fraud_mini100k.csv')
            #dataset = pd.read_csv('./data/credit_card_transactions/fraudTest_mini.csv')

            st.header('About the dataset')

            about_data = """This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020.
            It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants. This subset of the dataset was created by randomly sampling from the original data."""
            st.markdown(about_data)


            st.link_button("Go to Data", "https://www.kaggle.com/datasets/kartik2112/fraud-detection/data")

            st.header('Sample Raw Data')
            st.write(dataset.head(10))
            #st.write(dataset.columns)
            
            basicEDA2(dataset, '100k')


        elif data_option == "Upload Dataset - CSV":
            st.markdown(" Feature releasing soon! Please choose from other options!")
    
