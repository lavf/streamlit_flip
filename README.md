# Data Science - Project "Flip The Script"
From our first meeting, I could better understand our target and what we wanted from Flip-The-Script. On the same day I started developing a function and doing some research. There were some algorithms related to the subject but as I thought, they covered part of our MVP and not all the requirements.

RegEx was the key in the functions, therefore, I had to learn it. After the first implementations, I realized that the program ran faster and without any problem. Basically the function used a long string as a dictionary and by every match with the text (user input), the function delivered the conversion. This applied morphological and orthographic distinctions, and provided a conversion in accordance to these rules.

The next day I could deliver screenshots to show that it was possible to create a gender converter so with that the other members could start doing their tasks. My first steps in this project were as a Software Developer more than as a Data Scientist.

Following the same line, I started developing the neutral converter and at the same time made the gender converter robuster. Since I took the lead in Software development, a project model was necessary to avoid any inconsistencies. As a beginner, the "Trial and Error" model was the best choice to be implemented and Kanban list was the tool that we used as a group. In order to define my milestones and to show what I was currently doing, I actively used it.

The way of thinking for the neutral converter was different. A switch function was not an option because the conversion didn’t require a reversed feature. Thus, I created a dictionary and developed a function that by every match with the text (user input) delivered a conversion. The complexity of the neutral converter lied in the units, some of the conversion had more than one word so a RegEx pattern was not the best option to be implemented.

Having both converters, the module test phase was the next step. My idea was to use it as a reference for the dictionary articles from well-known news media. From the first samples, the first bugs came to light. There was an issue when the program converted the pronoun "he" or "she" into "they". The verb, which follows, remains the same (3rd person singular). If I would have tried to change every verb, I would have had to create a dictionary with every converted verb.

With the neutral converter, I thought about how to create a data analysis using articles. I found a dataset on Kaggle (article_news.csv) and started using it to get some insights. Nevertheless, the quantity of words of the title and headlines were not enough to give them. Therefore, I started analyzing news articles with at least 200 words and recorded them in my own dataset. I noticed that the most insightful article was a particular case about a person (topic: local news). News articles about politics or health (Covid-19) do not use feminine or masculine articles or pronouns to a great extent.

By the end of December, there was a beta version for gender converter (adding new words) and a beta version with disclaimer (verbs - morphological rules) for neutral converter. We decided as a group to not push the dataset on GitHub due to legal terms and added another attribute called "gender_author" to the dataset in order to make a distinction between male and female authors. I set a milestone, that by January 17 a hundred analyzed articles should be included in the dataset. In connection with Linguistics, I started documenting the most significant conversions included in the dictionaries so this page should be included as "References" on the website.

In early January I defined the three main attributes, which should be used for prediction (Machine Learning):

*Sources: 'abc news', 'al jazeera', 'bbc news', 'cnbc', 'cnn', 'deutsche welle', 'newsweek', 'npr', 'reuters', 'science', 'the irish times', 'the new york times'*

*Topics: 'business', 'culture', 'food and drinks', 'health', 'local news', 'people', 'politics', 'social science', 'sports', 'technology', 'travel', 'world'*

*Author's Gender: 'male', 'female', 'none'*

I used the Counter library, created another functions for text analysis and categorized the units, which were converted, grammatically into ten groups:

1. *'feminine personal pronoun (she)',*
2. *'masculine personal pronoun (he)',*
3. *'feminine determiners (her, herself)',*
4. *'masculine object pronouns and determiners (him, his, himself)',*
5. *'masculine nouns',*
6. *'feminine nouns',*
7. *'adjectives with masculine connotation',*
8. *'adjectives with feminine connotation',*
9. *'masculine title',*
10. *'feminine title'.*

I registered in the dataset: the number of words per article, the number of words converted, the number of words converted as per grammatical and morphological category, a list with the words converted, and six lists with words converted to analyze possible bias according to the following categories: *'masculine nouns', 'feminine nouns', 'adjectives with masculine connotation', 'adjectives with feminine connotation', 'masculine title', 'feminine title'*.

After reaching 100 analyzed articles, I started plotting. For this purpose, I used Streamlit in order to see all the data and get insights. With the first plots, I explained the insights to front end since they were in charge of plotting graphics for the website. However, the first insights were used as a reference since the 100 articles represented only 37% of total analysis.

In late January I completed the analysis of 276 articles, the dataset was done. In this way, I could finish my plots for machine learning and some of them are similar to the following ones:

   ![plots](https://user-images.githubusercontent.com/73216174/107207077-264c2200-6a00-11eb-81c4-570a762e7bad.png)

The next step was to develop a prediction using an algorithm from scikitlearn. Due to the nature of the dataset "categorical data", I decided on using KNeighbors. All the steps I took were through troubleshooting. I created another csv file, where I labeled every article according to some parameters that I established. Mean() was the key function for every calculation and after getting these values, I crossed information according to three attributes: Topic, Source and Global mean (all articles). The labels were established as below:

*Non-biased: Equal to 0,*

*Slightly biased: More than 0 and less than the mean value,*

*Biased: Equal to or more than the mean value.*

After comparing the three labels from the three aforementioned attributes, I assigned one unique label, which summarized the other three and was the most frequently occurring label.

Having this labeled csv file, I got many errors related to the type of data when creating the model. Then I used KNeighbors from scratch (euclidean distance) and successfully got predictions. As our mentor recommended me, I tried once again with scikitlearn and I could successfully get predictions with the following scores:

*Accuracy of test set: 0.51*

*Accuracy of training set: 0.41*

*Accuracy of set: 0.48*

The dataset didn’t have enough attributes in order to increase accuracy. At the beginning, my model had a 63% accuracy without splitting data into train and test. Nevertheless, when I talked about the model in the weekly meeting, as a group we decided to add two extra options for author's gender: "non-binary" and "I don't know", which were summarized in "none". There are few articles in the dataset, which don't include gender. In this way, the model must predict data that doesn't exist in the dataset. Splitting was a must. I tuned the model with: StandardScale(), GridSearchCV(), which resulted in:

*Best leaf_size: 1*

*Best p: 1*

*Best n_neighbors: 15*

Finally I used the library Pickle in order to save the model as a sav file and implement it in production.

The estimates delivered by the algorithm should be used as a reference since the dataset is not large enough to set a precedent. The research consisting of: 276 articles from 12 sources about 12 topics most of them published between December 2020 and the first week of February 2021, can give an insight that gendered language does exist in the media and varies according to source, topic and author's gender. Future studies and extension of the dataset will help the algorithm to be robust and a link between the neutral converter and the model could provide a direct bias prediction with higher accuracy using units converted and categories.

These softwares together with the algorithm were intended to encourage users to reflect about these questions.

Gender converter:

* Does the reversed text show a difference in relation to the original one? Does the meaning change? (semantic differences)

Neutral converter:

* Does the meaning of the text change when using the software? (semantic and morphosyntactic differences)
* Is it necessary to define the gender of a person through pronouns, nouns, determiners, titles, etc. in a text?
* Is it necessary to define the gender of an expression through adverbs or adjectives in a text?

Model:

* Is there a reason why the media use certain words instead of neutral options?
* Do topics or the author's gender have an influence on how articles are written?

[Data Science](#data-science) </br>
