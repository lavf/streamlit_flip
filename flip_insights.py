import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import statistics
import matplotlib.pyplot as plt
from knn_from_scratch import knn, euclidean_distance
import pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

plt.style.use('seaborn')

news = pd.read_csv("dataset_flip.csv", index_col=0)

st.title(":alembic: Data Science Report")
st.write("## __*Project: 'Flip the Script'*__")
st.write("### Behind the scenes —*from Data Analysis to Machine Learning*")
st.write("[Leticia Valladares](http://leticiavalladares.me)")
st.write("\n")
st.write("\n")
st.write("### Prediction")
st.write("\n")
st.write("Select **source**, **topic** and **author's gender** from the sidebar to predict if an article under these parameters uses gendered language.")

st.sidebar.header("Prediction parameters")
st.sidebar.header(":crystal_ball:")

src = {  "abc news": 1,
        "al jazeera": 2,
        "bbc news": 3,
        "cnn": 4,
        "deutsche welle": 5,
        "newsweek": 6,
        "reuters": 7,
        "the irish times": 8,
        "the new york times": 9,
        "science": 10,
        "cnbc": 11,
        "npr": 12        
        }

tpc= {  "business": 1,
        "culture": 2,
        "food and drinks": 3,
        "health": 4,
        "local news": 5,
        "people": 6,
        "politics": 7,
        "sports": 8,
        "technology": 9,
        "travel": 10,
        "world": 11,
        "social science": 12        
        }

sex= {  "male": 0,
        "female": 1,
        "none": 2      
        }

def user_input_features():
    source_name = st.sidebar.selectbox("Source", ('abc news', 'al jazeera', 'bbc news', 'cnbc', 'cnn', 'deutsche welle', 'newsweek', 'npr', 'reuters', 'science', 'the irish times', 'the new york times'))
    topic = st.sidebar.selectbox("Topic", ('business', 'culture', 'food and drinks', 'health', 'local news', 'people', 'politics', 'social science', 'sports', 'technology', 'travel', 'world'))
    gender_author = st.sidebar.selectbox("Author's gender", ('female', 'male', 'none'))
#    #words_count = st.sidebar.slider("Quantity of words", 279, 2632, 686)
    features = [src[source_name], tpc[topic], sex[gender_author]]
    features = np.array(features)
    # data = {"source_name": source_name,
    #         "topic": topic,
    #         "gender_author": gender_author}
    # features = pd.DataFrame(data, index=[0])
    return features

#st.subheader("User Input parameters")
user_input0 = user_input_features()
#st.write(user_input)

user_input1 = np.array(user_input0).reshape(1,-1)
user_input2 = np.array(user_input0).reshape(-1,1)

bias= { 0 :"non-biased",
        1: "slightly biased",
        2: "biased"      
        }

y_prediction = loaded_model.predict(user_input1)
prediction = y_prediction[0]

st.write("\n")
st.write("\n")
st.write("#### :crystal_ball: Results")
st.write("\n")
#Prediction as a string
st.write("*The article might be: *" + str(bias[prediction]))
st.write("\n")
st.write("\n")


st.write("### Table of Contents")
st.write("\n")
st.write("1. Introduction")
st.write("2. Software Algorithm")
st.write("3. Dataset")
st.write("4. Machine Learning")
st.write("5. Plots")
st.write("6. Conclusion")
st.write("7. References")
st.write("\n")
st.write("\n")

st.write("### 1. Introduction")
st.write("\n")
st.write("This research consisted of a morphosyntactic and semantic analysis of 276 articles written in English with the purpose of recognizing the existence of gendered language in well-known news media. To summarize and analize observations, I used summary statistics to create a prediction model and as a reference for this report, added euclidean distance prediction with three neighbors.")
st.write("\n")
st.write("#### :crystal_ball: Prediction according to euclidean distance")
st.write("\n")
st.write("Select **source**, **topic** and **author's gender** from the sidebar to see prediction.")
st.write("\n")

def predict_news(news_query, k_predictions):
    raw_news_data = []
    with open(r'news_bias.csv') as md:
        next(md)

        for line in md.readlines():
            data_row = line.strip().split(',')
            raw_news_data.append(data_row)

    news_bias = []
    for row in raw_news_data:
        data_row = list(map(float, row[2:]))
        news_bias.append(data_row)

    prediction_indices, _ = knn(
        news_bias, news_query, k=k_predictions,
        distance_fn=euclidean_distance, choice_fn=lambda x: None
    )

    news_predictions = []
    for _, index in prediction_indices:
        news_predictions.append(raw_news_data[index])

    return news_predictions

if __name__ == '__main__':
    predicted_case = predict_news(news_query=user_input2, k_predictions=3)

    # Print suggested predictions
    for recommendation in predicted_case:
        st.write("*33.3% chance *" + str(recommendation[1]))


st.write("\n")
st.write("\n")
st.write("### 2. Software Algorithm")
st.write("\n")

st.write("#### 2.1. Original Words vs. Converted Words (replaced with neutral alternatives)")
fig,ax = plt.subplots() #must create a subplot
ax = plt.plot(news['words_count'], color='mediumaquamarine', label='Words in total')
plt.plot(news['words_converted'], color='mediumorchid', label='Words converted')
plt.ylabel("Quantity of words")
plt.xlabel('Quantity of articles')
plt.legend(loc='upper left')
#plt.ylim(200, 1000)
#ax = sns.lineplot(x=news['words_count'], y=news['words_converted'], data= news['topic']) #, color='indianred'
st.pyplot(fig)
st.write("After creating the gender and neutral converters, I started with text analysis. The first samples gave me an idea on how to structure this study. I defined the three main attributes, which were thought for machine learning:")

st.write("*Sources: 'abc news', 'al jazeera', 'bbc news', 'cnbc', 'cnn', 'deutsche welle', 'newsweek', 'npr', 'reuters', 'science', 'the irish times', 'the new york times'*")

st.write("*Topics: 'business', 'culture', 'food and drinks', 'health', 'local news', 'people', 'politics', 'social science', 'sports', 'technology', 'travel', 'world'*")

st.write("*Author's Gender: 'male', 'female', 'none'*")

st.write("I used the Counter library, created another functions for text analysis and categorized the units, which were converted, grammatically into ten groups:")

st.write("1. *'female personal pronoun (she)',*")
st.write("2. *'male personal pronoun (he)',*")
st.write("3. *'feminine determiners (her, herself)',*")
st.write("4. *'masculine determiners (him, his, himself)',*")
st.write("5. *'masculine nouns',*")
st.write("6. *'femenine nouns',*")
st.write("7. *'adjectives with masculine connotation',*")
st.write("8. *'adjectives with femenine connotation',*")
st.write("9. *'male title',*")
st.write("10. *'female title'.*")

st.write("I registered in the dataset: the number of words per article, the number of words converted, the number of words converted as per grammatical and morphological category, a list with the words converted, and six lists with words converted to analyze possible bias according to the following categories: *'masculine nouns', 'femenine nouns', 'adjectives with masculine connotation', 'adjectives with femenine connotation', 'male title', 'female title'*.")

st.write("\n")
st.write("\n")
st.write("### 3. Dataset")
st.write("\n")
st.write("After doing some research and visiting popular websites used by data scientists, I realized that the existing datasets were out of scope of our MVP. I had no other choice but to create a dataset from scratch using the algorithm, that I created for the app.")

col_names = news.columns.tolist()
col_select = st.multiselect("Columns",col_names, default = col_names)
st.dataframe(news[col_select])
st.write("\n")
st.write("\n")

st.write("### 4. Machine Learning")
st.write("\n")

st.write("#### 4.1. Residual of a regression")
fig,ax = plt.subplots()
ax = sns.residplot(data=news, x="words_converted", y="words_count", color="indianred")
st.pyplot(fig)

st.write("#### 4.2. Higher-order regressions")
fig,ax = plt.subplots()
ax = plt.scatter(news['words_converted'], news['words_count'], label='Articles', color='#766BCE', marker='o')
sns.regplot(data=news, x="words_converted", y="words_count", color="#E3ABCA", scatter=None, label= "First order")
sns.regplot(data=news, x="words_converted", y="words_count", color="#94DBBC", scatter=None, order=2, label= "Second order")
plt.legend(loc='upper right')
st.pyplot(fig)

st.write("Due to the nature of the dataset (categorical dataset), I decided to use KNeighbors. Likewise, I created another csv file, where I labeled every article according to some parameters that I established. These parameters were mainly set by calculation the mean and after getting values, I crosschecked information according to three attributes: **topic**, **source** and **global mean** (all articles). The labels were established as below:")
st.write("\n")
st.write("* *Non-biased: Equal to 0,*")
st.write("* *Slightly biased: More than 0 and less than the mean value,*")
st.write("* *Biased: Equal to or more than the mean value.*")
st.write("\n")
st.write("After comparing the three labels from the three aforementioned attributes, I assigned one unique label, which summarized the other three and was the most frequently occurring label.")
#st.write("There is a high chance that the article is " + str(recommendation[1]))
st.write("The KNN model delivered the following scores:")
st.write("\n")
st.write("*Accuracy of test set: 0.51*")

st.write("*Accuracy of training set: 0.41*")

st.write("*Accuracy of set: 0.48*")
st.write("\n")
st.write("The dataset doesn’t have enough attributes in order to increase accuracy. At the beginning, the model had a 63% accuracy without splitting data into train and test. I tuned the model with: StandardScale(), GridSearchCV(), which resulted in:")
st.write("\n")
st.write("*Best leaf_size: 1*")
st.write("*Best p: 1*")
st.write("*Best n_neighbors: 15*")
st.write("\n")


total_pct = {'Category': ['male personal pronoun','masculine determiners','female personal pronoun','masculine nouns','feminine determiners','feminine nouns','male titles','adjectives with femenine connotation','female titles','adjectives with masculine connotation'], 
            'Percentage':[2.9,2.1,1.5,1.1,0.9,0.9,0.3,0.1,0.1,0]} 

total_fem_noun = {'Words': ['women', 'woman', 'mother', 'wife', 'daughter', 'Women', 'sisters', 'spokeswoman', 'girl','girls'],
                'Quantity of articles':[58,28,23,19,17,11,9,7,7,7]}

total_masc_noun = {'Words': ['man', 'men', 'father', 'son', 'husband','spokesman','brother','guy','actor','boy'],
                   'Quantity of articles':[37,37,17,15,15,11,11,10,8,6] }

total_adj_fem = {'Words': ['female', 'feminist', 'emotional', 'feminine', 'Female', 'Emotional', 'femininity'],
                'Quantity of articles':[16,5,5,3,2,1,1]}

total_adj_masc = {'Words': ['male','masculine', 'Male', 'masculinity'],
                'Quantity of articles':[11,4,3,1]}

total_fem_title ={'Words': ['Ms','Lady'],
                'Quantity of articles':[13,6]}

total_masc_title ={'Words':['Mr','Lord','Sir'],
                'Quantity of articles':[28,3,1]}

df_sum = pd.DataFrame(total_pct) 
df_fem_noun = pd.DataFrame(total_fem_noun) 
df_masc_noun = pd.DataFrame(total_masc_noun)
df_adj_fem = pd.DataFrame(total_adj_fem)
df_adj_masc = pd.DataFrame(total_adj_masc)
df_fem_title = pd.DataFrame(total_fem_title)
df_masc_title = pd.DataFrame(total_masc_title)

#st.dataframe(df_sum)

st.write("#### 4.3. Global Statistics")

#st.subheader("Global Analysis / Conversion ratio - Min")
news_ratio = news[news['ratio'] > 0]
st.write()

st.write(pd.DataFrame({
     'Measure': ["Global Analysis / Converted Words - Mean", "Global Analysis / Quantity of Words - Mean", "Global Analysis / Conversion ratio - Max", "Global Analysis / Conversion ratio - Mean", "Global Analysis: Femenine nouns / Ratio % - Mean", "Global Analysis: Masculine nouns / Ratio % - Mean", "Global Analysis / Conversion ratio - Min"],
     'Values': [news['words_converted'].mean(), news['words_count'].mean(), news['ratio'].max(), news['ratio'].mean(), news['rt_noun_fem'].mean(), news['rt_noun_masc'].mean(), news_ratio['ratio'].min()],
 }))

st.write("#### 4.4. Statistics - Source: Deutsche Welle")

news_source_name = news[news['source_name'] =='deutsche welle']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write()

news_source_name = news[news['source_name'] =='deutsche welle']
b = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]

news_source_name = news[news['source_name'] =='deutsche welle']
c = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]

st.write(pd.DataFrame({
     'Measure': ["Nouns / Ratio % - Mean", "Adjectives / Ratio % - Mean", "Both / Ratio % - Mean"],
     'Values': [statistics.mean(a),statistics.mean(b),statistics.mean(c)],
}))

st.write("#### 4.5. Statistics - Source: NPR")

#st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='npr']
d = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
#st.write(statistics.mean(d))

#st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='npr']
e = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
#st.write(statistics.mean(e))

#st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='npr']
f = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
#st.write(statistics.mean(f))

st.write(pd.DataFrame({
     'Measure': ["Nouns / Ratio % - Mean", "Adjectives / Ratio % - Mean", "Both / Ratio % - Mean"],
     'Values': [statistics.mean(d),statistics.mean(e),statistics.mean(f)],
}))

st.write("#### 4.6. Statistics - Source: BBC News")

#st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='bbc news']
g = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
#st.write(statistics.mean(g))

#st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='bbc news']
h = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
#st.write(statistics.mean(h))

#st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='bbc news']
i = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
#st.write(statistics.mean(i))

st.write(pd.DataFrame({
     'Measure': ["Nouns / Ratio % - Mean", "Adjectives / Ratio % - Mean", "Both / Ratio % - Mean"],
     'Values': [statistics.mean(g),statistics.mean(h),statistics.mean(i)],
}))

st.write("#### 4.7. Statistics - Source: The Irish Times")

#st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='the irish times']
j = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
#st.write(statistics.mean(j))

#st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='the irish times']
k = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
#st.write(statistics.mean(k))

#st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='the irish times']
l = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
#st.write(statistics.mean(l))

st.write(pd.DataFrame({
     'Measure': ["Nouns / Ratio % - Mean", "Adjectives / Ratio % - Mean", "Both / Ratio % - Mean"],
     'Values': [statistics.mean(j),statistics.mean(k),statistics.mean(l)],
}))


st.write("\n")
st.write("\n")
st.write("### 5. Plots")
st.write("### 5.1. Plots - Quantitative Analysis")

# st.write("### Converted words per category")
# fig,ax = plt.subplots() #must create a subplot
# ax = plt.plot(news['noun_masc'], color='mediumaquamarine', label='Masc nouns')
# plt.plot(news['noun_fem'], color='mediumorchid', label='Fem nouns')
# plt.plot(news['deter_pron_masc'], color='limegreen', label='Masc determiners')
# plt.plot(news['deter_pron_fem'], color='hotpink', label='Fem determiners')
# plt.ylabel("Quantity of words")
# plt.xlabel('Quantity of articles')
# plt.legend(loc='upper left')
# st.pyplot(fig)

st.write("#### 5.1.1. Categories in %")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=df_sum['Percentage'], y=df_sum['Category'], alpha=0.5, palette="cubehelix", ci=None) #rocket, hls 
#plt.ylim(1.201, 2.303)
#plt.xlim(0, 30)
plt.ylabel('Converted words ratio')

plt.title('Categories in %')
plt.legend(loc='upper right')

total = len(df_sum['Percentage'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
st.pyplot(fig)

st.write("#### 5.1.2. The 10 Most Frequently Used Femenine Nouns")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=df_fem_noun['Quantity of articles'], y=df_fem_noun['Words'], alpha=0.5, palette="viridis", ci=None) #rocket, hls 
#plt.ylim(1.201, 2.303)
#plt.xlim(0, 10)
plt.ylabel('Femenine Nouns')
plt.xlabel("Quantity of articles, where these nouns were found")
plt.title('The 10 Most Frequently Used Femenine Nouns')
#plt.legend(loc='upper right')
st.pyplot(fig)

st.write("#### 5.1.3. The 10 Most Frequently Used Masculine Nouns")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=df_masc_noun['Quantity of articles'], y=df_masc_noun['Words'], alpha=0.5, palette="crest", ci=None) #rocket, hls 
#plt.ylim(1.201, 2.303)
#plt.xlim(0, 10)
plt.ylabel('Masculine Nouns')
plt.xlabel("Quantity of articles, where these nouns were found")
plt.title('The 10 Most Frequently Used Masculine Nouns')
#plt.legend(loc='upper right')
st.pyplot(fig)

st.write("#### 5.1.4. Adjectives with femenine connotation in articles")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=df_adj_fem['Quantity of articles'], y=df_adj_fem['Words'], alpha=0.5, palette="mako", ci=None) #rocket, hls 
#plt.ylim(1.201, 2.303)
#plt.xlim(0, 10)
plt.ylabel('Adjectives with femenine connotation')
plt.xlabel("Quantity of articles, where these adjectives were found")
plt.title('Adjectives with femenine connotation in articles')
#plt.legend(loc='upper right')
st.pyplot(fig)

st.write("#### 5.1.5. Adjectives with masculine connotation in articles")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=df_adj_masc['Quantity of articles'], y=df_adj_masc['Words'], alpha=0.5, palette="flare", ci=None) #rocket, hls 
#plt.ylim(1.201, 2.303)
#plt.xlim(0, 10)
plt.ylabel('Adjectives with masculine connotation')
plt.xlabel("Quantity of articles, where these adjectives were found")
plt.title('Adjectives with masculine connotation in articles')
#plt.legend(loc='upper right')
st.pyplot(fig)

st.write("#### 5.1.6. Female titles in articles")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=df_fem_title['Quantity of articles'], y=df_fem_title['Words'], alpha=0.5, palette="magma", ci=None) #rocket, hls 
#plt.ylim(1.201, 2.303)
#plt.xlim(0, 10)
plt.ylabel('Female titles')
plt.xlabel("Quantity of articles, where these titles were found")
plt.title('Female titles in articles')
#plt.legend(loc='upper right')
st.pyplot(fig)

st.write("#### 5.1.7. Masculine titles in articles")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=df_masc_title['Quantity of articles'], y=df_masc_title['Words'], alpha=0.5, palette="Set2", ci=None) #rocket, hls 
#plt.ylim(1.201, 2.303)
#plt.xlim(0, 10)
plt.ylabel('Masculine titles')
plt.xlabel("Quantity of articles, where these titles were found")
plt.title('Masculine titles in articles')
#plt.legend(loc='upper right')
st.pyplot(fig)

st.write("\n")
st.write("\n")
st.write("### 5.2. Plots using ratio")

st.write("#### 5.2.1. Author's gender")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=news['gender_author'], y=news['ratio'], alpha=0.5, hue=news['gender_author'], palette="hls", ci=None) #rocket, hls 
plt.ylim(1.201, 2.303)
#plt.xlim(0, 10)
plt.ylabel('Converted words ratio')
plt.xlabel("Author's gender")
plt.title('Ratio = Neutral * % / Original')
plt.legend(loc='upper right')
st.pyplot(fig)

st.write("#### 5.2.2. Author's gender per topic")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=news['ratio'], y=news['topic'], alpha=0.5, hue=news['gender_author'], palette="hls", ci=None, order=[ 'people', 'culture', 'local news', 'sports', 'social science', 'world', 'politics', 'business', 'technology', 'food and drinks', 'travel','health']) #rocket, hls 
#plt.ylim(0, 4)
#plt.xlim(0, 10)
plt.ylabel('Topic')
plt.xlabel('Converted words ratio')
plt.title("Author's gender per topic")
plt.legend(loc='lower right')
st.pyplot(fig)

st.write("#### 5.2.3. Global analysis - Topic")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=news['ratio'], y=news['topic'], alpha=0.5, palette="hls", ci=None, order=[ 'people', 'culture', 'local news', 'sports', 'social science', 'world', 'politics', 'business', 'technology', 'food and drinks', 'travel','health']) #rocket, hls 
#plt.ylim(0, 4)
#plt.xlim(0, 10)
plt.ylabel('Topic')
plt.xlabel('Converted words ratio')
plt.title('Global analysis - Topic')
st.pyplot(fig)

st.write("#### 5.2.4. Author's gender per source")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=news['ratio'], y=news['source_name'], alpha=0.5, hue=news['gender_author'], palette="hls", ci=None, order=['the new york times', 'the irish times', 'bbc news', 'newsweek', 'deutsche welle','cnbc','cnn', 'npr','al jazeera', 'reuters','abc news', 'science']) #rocket, hls 
#plt.ylim(0, 4)
#plt.xlim(0, 10)
plt.ylabel('Source')
plt.xlabel('Converted words ratio')
plt.title('Ratio = Neutral * % / Original')
plt.legend(loc='lower right')
st.pyplot(fig)

st.write("#### 5.2.5. Global analysis - Source")
fig,ax = plt.subplots() #must create a subplot
ax =sns.barplot(x=news['ratio'], y=news['source_name'], alpha=0.5, palette="hls", ci=None, order=['the new york times', 'the irish times', 'bbc news', 'newsweek', 'deutsche welle','cnbc','cnn', 'npr','al jazeera', 'reuters','abc news', 'science']) #rocket, hls 
#plt.ylim(0, 4)
#plt.xlim(0, 10)
plt.ylabel('Source')
plt.xlabel('Converted words ratio')
plt.title('Global analysis - Source')
st.pyplot(fig)

st.write("#### 5.2.6. Ratio of conversions per topic")
fig,ax = plt.subplots() #must create a subplot
ax =sns.scatterplot(x=news['ratio'], y=news['topic'], alpha=0.5, hue=news['words_count'], size=news['words_count'], sizes=(50,500), palette="mako") #rocket, hls 
#plt.ylim(0, 10)
plt.xlim(0, 10)
plt.ylabel("Topics")
plt.xlabel('Converted words ratio')
plt.title('Ratio = Neutral * % / Original')
plt.legend(loc='lower right')
st.pyplot(fig)

st.write("#### 5.2.7. Ratio of conversions per source")
fig,ax = plt.subplots() #must create a subplot
ax =sns.scatterplot(x=news['ratio'], y=news['source_name'], alpha=0.5, hue=news['words_count'], size=news['words_count'], sizes=(50,500), palette="rocket") #rocket, hls 
#plt.ylim(0, 10)
plt.xlim(0, 10)
plt.ylabel("Source")
plt.xlabel('Converted words ratio')
plt.title('Ratio = Neutral * % / Original')
plt.legend(loc='lower right')
st.pyplot(fig)

st.write("#### 5.2.8. Male nouns")
fig,ax = plt.subplots() #must create a subplot
ax =sns.scatterplot(x=news['rt_noun_masc'], y=news['topic'], hue=news['gender_author'], palette="husl") #rocket, hls 
#plt.ylim(0, 10)
#plt.xlim(0, 10)
plt.ylabel("Topic")
plt.xlabel('Quantity of male nouns (%) from total converted words')
plt.title('')
plt.legend(loc='lower right')
st.pyplot(fig)

st.write("#### 5.2.9. Female nouns")
fig,ax = plt.subplots() #must create a subplot
ax =sns.scatterplot(x=news['rt_noun_fem'], y=news['topic'], hue=news['gender_author'], palette="husl") #rocket, hls 
#plt.ylim(0, 10)
#plt.xlim(0, 10)
plt.ylabel("Topic")
plt.xlabel('Quantity of female nouns (%) from total converted words')
plt.title('')
plt.legend(loc='lower right')
st.pyplot(fig)

st.write("\n")
st.write("\n")
st.subheader("Bias - ABC News")

st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='abc news']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='abc news']
a = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='abc news']
a = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Bias - CNBC")

st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='cnbc']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='cnbc']
a = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='cnbc']
a = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Bias - The New York Times")

st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='the new york times']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='the new york times']
a = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='the new york times']
a = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Bias - Al Jazeera")

st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='al jazeera']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='al jazeera']
a = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='al jazeera']
a = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Bias - Science")

st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='science']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='science']
a = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='science']
a = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Bias - CNN")

st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='cnn']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='cnn']
a = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='cnn']
a = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Bias - Reuters")

st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='reuters']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='reuters']
a = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='reuters']
a = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Bias - Newsweek")

st.subheader("Nouns / Ratio % - Mean")
news_source_name = news[news['source_name'] =='newsweek']
a = [news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_source_name = news[news['source_name'] =='newsweek']
a = [news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_source_name = news[news['source_name'] =='newsweek']
a = [news_source_name['rt_title_fem'].mean(), news_source_name['rt_title_masc'].mean(),news_source_name['rt_noun_fem'].mean(), news_source_name['rt_noun_masc'].mean(), news_source_name['rt_adj_conn_fem'].mean(), news_source_name['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Bias - Social Science")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='social science']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='social science']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='social science']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Business")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='business']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='business']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='business']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Sports")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='sports']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='sports']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='sports']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Food and Drinks")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='food and drinks']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='food and drinks']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='food and drinks']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Technology")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='technology']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='technology']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='technology']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Health")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='health']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='health']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='health']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Politics")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='politics']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='politics']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='politics']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Travel")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='travel']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='travel']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='travel']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - People")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='people']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='people']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='people']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Local News")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='local news']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='local news']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='local news']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - Culture")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='culture']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='culture']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='culture']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.write("## Bias - World")

st.subheader("Nouns / Ratio % - Mean")
news_topic = news[news['topic'] =='world']
a = [news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Adjectives / Ratio % - Mean")
news_topic = news[news['topic'] =='world']
a = [news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

st.subheader("Both / Ratio % - Mean")
news_topic = news[news['topic'] =='world']
a = [news_topic['rt_title_fem'].mean(), news_topic['rt_title_masc'].mean(),news_topic['rt_noun_fem'].mean(), news_topic['rt_noun_masc'].mean(), news_topic['rt_adj_conn_fem'].mean(), news_topic['rt_adj_conn_masc'].mean()]
st.write(statistics.mean(a))

# st.write("# Social Science Stats")

# st.subheader("Converted Words - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['words_converted'].mean())

# st.subheader("Converted Words - Min")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['words_converted'] >0]
# st.write(news_topic_min['words_converted'].min())

# st.subheader("Converted Words - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['words_converted'].max())

# st.subheader("Ratio % - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['ratio'].mean())

# st.subheader("Ratio % - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['ratio'] >0]
# st.write(news_topic_min['ratio'].min())

# st.subheader("Ratio % - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['ratio'].max())

# st.subheader("Femenine nouns / Words - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['noun_fem'].mean())

# st.subheader("Femenine nouns / Words - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['noun_fem'].max())

# st.subheader("Femenine nouns / Words - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['noun_fem'] >0]
# st.write(news_topic_min['noun_fem'].min())

# st.subheader("Femenine nouns / Ratio % - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['rt_noun_fem'].mean())

# st.subheader("Femenine nouns / Ratio % - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['rt_noun_fem'].max())

# st.subheader("Femenine nouns / Ratio % - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['rt_noun_fem'] >0]
# st.write(news_topic_min['rt_noun_fem'].min())

# st.subheader("Masculine nouns / Words - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['noun_masc'].mean())

# st.subheader("Masculine nouns / Words - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['noun_masc'].max())

# st.subheader("Masculine nouns / Words - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['noun_masc'] >0]
# st.write(news_topic_min['noun_masc'].min())

# st.subheader("Masculine nouns / Ratio % - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['rt_noun_masc'].mean())

# st.subheader("Masculine nouns / Ratio % - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['rt_noun_masc'].max())

# st.subheader("Masculine nouns / Ratio % - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['rt_noun_masc'] >0]
# st.write(news_topic_min['rt_noun_masc'].min())

# st.subheader("Adjectives with femenine connotation / Words - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['adj_conn_fem'].mean())

# st.subheader("Adjectives with femenine connotation / Words - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['adj_conn_fem'].max())

# st.subheader("Adjectives with femenine connotation / Words - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['adj_conn_fem'] >0]
# st.write(news_topic_min['adj_conn_fem'].min())

# st.subheader("Adjectives with femenine connotation / Ratio % - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['rt_adj_conn_fem'].mean())

# st.subheader("Adjectives with femenine connotation / Ratio % - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['rt_adj_conn_fem'].max())

# st.subheader("Adjectives with femenine connotation / Ratio % - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['rt_adj_conn_fem'] >0]
# st.write(news_topic_min['rt_adj_conn_fem'].min())

# st.subheader("Adjectives with masculine connotation / Words - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['adj_conn_masc'].mean())

# st.subheader("Adjectives with masculine connotation / Words - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['adj_conn_masc'].max())

# st.subheader("Adjectives with masculine connotation / Words - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['adj_conn_masc'] >0]
# st.write(news_topic_min['adj_conn_masc'].min())

# st.subheader("Adjectives with masculine connotation / Ratio % - Mean")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['rt_adj_conn_masc'].mean())

# st.subheader("Adjectives with masculine connotation / Ratio % - Max")
# news_topic = news[news['topic'] =='social science']
# st.write(news_topic['rt_adj_conn_masc'].max())

# st.subheader("Adjectives with masculine connotation / Ratio % - Min excluding zero")
# news_topic = news[news['topic'] =='social science']
# news_topic_min = news_topic[news_topic['rt_adj_conn_masc'] >0]
# st.write(news_topic_min['rt_adj_conn_masc'].min())

# st.write("# Technology Stats")

# st.subheader("Converted Words - Mean")
# news_topic = news[news['topic'] =='technology']
# st.write(news_topic['words_converted'].mean())

# st.subheader("Converted Words - Min")
# news_topic = news[news['topic'] =='technology']
# news_topic_min = news_topic[news_topic['words_converted'] >0]
# st.write(news_topic_min['words_converted'].min())

# st.subheader("Converted Words - Max")
# news_topic = news[news['topic'] =='technology']
# st.write(news_topic['words_converted'].max())

# st.subheader("Ratio % - Mean")
# news_topic = news[news['topic'] =='technology']
# st.write(news_topic['ratio'].mean())

# st.subheader("Ratio % - Min")
# news_topic = news[news['topic'] =='technology']
# news_topic_min = news_topic[news_topic['ratio'] >0]
# st.write(news_topic_min['ratio'].min())

# st.subheader("Ratio % - Max")
# news_topic = news[news['topic'] =='technology']
# st.write(news_topic['ratio'].max())

# st.write("# People Stats")

# st.subheader("Converted Words - Mean")
# news_topic = news[news['topic'] =='people']
# st.write(news_topic['words_converted'].mean())

# st.subheader("Converted Words - Min")
# news_topic = news[news['topic'] =='people']
# news_topic_min = news_topic[news_topic['words_converted'] >0]
# st.write(news_topic_min['words_converted'].min())

# st.subheader("Converted Words - Max")
# news_topic = news[news['topic'] =='people']
# st.write(news_topic['words_converted'].max())

# st.subheader("Ratio % - Mean")
# news_topic = news[news['topic'] =='people']
# st.write(news_topic['ratio'].mean())

# st.subheader("Ratio % - Min")
# news_topic = news[news['topic'] =='people']
# news_topic_min = news_topic[news_topic['ratio'] >0]
# st.write(news_topic_min['ratio'].min())

# st.subheader("Ratio % - Max")
# news_topic = news[news['topic'] =='people']
# st.write(news_topic['ratio'].max())

# st.write("# Local News Stats")

# st.subheader("Converted Words - Mean")
# news_topic = news[news['topic'] =='local news']
# st.write(news_topic['words_converted'].mean())

# st.subheader("Converted Words - Min")
# news_topic = news[news['topic'] =='local news']
# news_topic_min = news_topic[news_topic['words_converted'] >0]
# st.write(news_topic_min['words_converted'].min())

# st.subheader("Converted Words - Max")
# news_topic = news[news['topic'] =='local news']
# st.write(news_topic['words_converted'].max())

# st.subheader("Ratio % - Mean")
# news_topic = news[news['topic'] =='local news']
# st.write(news_topic['ratio'].mean())

# st.subheader("Ratio % - Min")
# news_topic = news[news['topic'] =='local news']
# news_topic_min = news_topic[news_topic['ratio'] >0]
# st.write(news_topic_min['ratio'].min())

# st.subheader("Ratio % - Max")
# news_topic = news[news['topic'] =='local news']
# st.write(news_topic['ratio'].max())


#news.groupby(['topic']).mean()
#news.groupby(['female']).mean()

st.write("\n")
st.write("\n")
st.write("### 5. Conclusion")
st.write("\n")
st.write("The estimates delivered by the algorithm should be used as a reference since the dataset is not large enough to set a precedent. The research consisting of: 276 articles from 12 sources about 12 topics most of them published between December 2020 and the first week of February 2021, can give an insight that gendered language does exist in the media and varies according to source, topic and author's gender. Future studies and extension of the dataset will help the algorithm to be robust and a link between the neutral converter and the model could provide a direct bias prediction with higher accuracy using units converted and categories.")

st.write("These softwares together with the algorithm were intended to encourage users to reflect about these questions.")
st.write("\n")
st.write("Gender converter:")
st.write("\n")
st.write("* Does the reversed text show a difference in relation to the original one? Does the meaning change? (semantic differences)")
st.write("\n")
st.write("Neutral converter:")
st.write("\n")
st.write("* Does the meaning of the text change when using the software? (semantic and morphosyntactic differences)")
st.write("* Is it necessary to define the gender of a person through pronouns, nouns, determiners, titles, etc. in a text?")
st.write("* Is it necessary to define the gender of an expression through adverbs or adjectives in a text?")
st.write("\n")
st.write("Model:")
st.write("\n")
st.write("* Is there a reason why the media use certain words instead of neutral options?")
st.write("* Do topics or the author's gender have an influence on how articles are written?")
st.write("\n")
st.write("\n")
st.write("### 6. References")
st.write("\n")
st.write("✔️ [same-sexparents.com: 'maddy' and related words for mommy and daddy](https://www.same-sexparents.com/post/gender-neutral-non-binary-parent-titles)")

st.write("✔️ [European Parliament: Gendered Neutral Language Guideline](https://www.europarl.europa.eu/cmsdata/151780/GNL_Guidelines_EN.pdf)")

st.write("✔️ [Cambridge Dictionary: Sexist Language](https://dictionary.cambridge.org/de/grammatik/britisch-grammatik/sexist-language)")

st.write("✔️ [Oxford Learners Dictionary: Gender Fluid](https://www.oxfordlearnersdictionaries.com/definition/english/gender-fluid)")

st.write("✔️ [Teen Vogue: How to use gender-neutral language](https://www.teenvogue.com/story/how-to-use-gender-neutral-words)")

st.write("✔️ [Vogue: “They” as a singular pronoun](https://www.vogue.com/article/they-nonbinary-pronoun-added-to-dictionary)")

st.write("✔️ [Merriam-Webster: “They” as a singular pronoun](https://www.merriam-webster.com/words-at-play/singular-nonbinary-they)")

st.write("✔️ [Grammarly: “They” as a singular pronoun](https://www.grammarly.com/blog/use-the-singular-they/)")

st.write("✔️ [Gender-Inclusive Language](https://writingcenter.unc.edu/tips-and-tools/gender-inclusive-language/)")

st.write("✔️ [Gender Neutral Aunt/Uncle](https://medium.com/bein-enby/gender-neutral-aunt-uncle-87041eccd765)")

st.write("✔️ [Gender Neutral: Paternalism ≠ Parentalism](https://opencommons.uconn.edu/cgi/viewcontent.cgi?article=1433&context=law_review])")

st.write("✔️ [Manslaughter: Legal term for a form of homicide has no gender-neutral replacement.](http://web.mit.edu/comdor/editguide/style-matters/gender_neutral.html)")

st.write("✔️ [Examples of common adjectives that carry a gender connotation and alternatives, and some examples of common gendered nouns and alternatives](https://eige.europa.eu/publications/toolkit-gender-sensitive-communication)")
st.write("\n")
st.write("\n")
st.write("© Copyright 2021 Leticia Valladares. All rights reserved.")
