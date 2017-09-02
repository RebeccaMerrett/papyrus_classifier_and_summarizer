# -*- coding: utf-8 -*-

import re
import nltk
nltk.download('punkt') #DOWNLOAD THIS ONCE WHEN FIRST RUNNING THE SCRIPT, THEN COMMENT OUT
nltk.download('averaged_perceptron_tagger') #DOWNLOAD THIS ONCE WHEN FIRST RUNNING THE SCRIPT, THEN COMMENT OUT
nltk.download('maxent_ne_chunker') #DOWNLOAD THIS ONCE WHEN FIRST RUNNING THE SCRIPT, THEN COMMENT OUT
nltk.download('words') #DOWNLOAD THIS ONCE WHEN FIRST RUNNING THE SCRIPT, THEN COMMENT OUT
from nltk import word_tokenize, pos_tag, ne_chunk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx

#To import the above:
#pip install nltk
#pip install scikit-learn
#pip install numpy
#pip install scipy
#pip install networkx

def function_1(text):
	paragraphs = text.split('\n\n')
	summary = []
	for i in paragraphs:
		#This finds each regex and counts the number of times it occurs in the paragraph. We'll use this to sum the weights of each regex.
		count_1 = len(re.findall('\$.*?million', i)) #Dollar figure (big figures, not small expenditure)
		count_2 = len(re.findall('\$.*?billion', i)) #Dollar figure
		count_3 = len(re.findall('\$.*?m', i)) #Dollar figure
		count_4 = len(re.findall('\$.*?bn', i)) #Dollar figure
		count_5 = len(re.findall('[2][0][1-9][0-9]', i)) #4-digits that indicate a year from 2011 to 2099
		count_6 = len(re.findall('\%', i)) #Percentage
		count_7 = len(re.findall('per cent', i)) #Percentage
		count_8 = len(re.findall('percent', i)) #Percentage
		count_9 = len(re.findall('completed', i)) #Cue word
		count_10 = len(re.findall('complete', i)) #Cue word
		count_11 = len(re.findall('completing', i)) #Cue word
		count_12 = len(re.findall('implement', i)) #Cue word
		count_13 = len(re.findall('implementing', i)) #Cue word
		count_14 = len(re.findall('due', i)) #Cue word
		count_15 = len(re.findall('commence', i)) #Cue word
		count_16 = len(re.findall('begin work', i)) #Cue word
		count_17 = len(re.findall('invested', i)) #Cue word
		count_18 = len(re.findall('investing', i)) #Cue word
		count_19 = len(re.findall('invest', i)) #Cue word
		count_20 = len(re.findall('surplus', i)) #Keyword
		count_21 = len(re.findall('budget', i)) #Keyword
		count_22 = len(re.findall('plan', i)) #Cue word
		count_23 = len(re.findall('planning', i)) #Cue word
		sum_1 = 1.2 * count_1
		sum_2 = 1.2 * count_2
		sum_3 = 1.2 * count_3
		sum_4 = 1.2 * count_4
		sum_5 = 0.8 * count_5
		sum_6 = 0.6 * count_6
		sum_7 = 0.6 * count_7
		sum_8 = 0.6 * count_8
		sum_9 = 0.5 * count_9
		sum_10 = 0.5 * count_10
		sum_11 = 0.5 * count_11
		sum_12 = 0.5 * count_12
		sum_13 = 0.5 * count_13
		sum_14 = 0.5 * count_14
		sum_15 = 0.5 * count_15
		sum_16 = 0.5 * count_16
		sum_17 = 0.5 * count_17
		sum_18 = 0.5 * count_18
		sum_19 = 0.5 * count_19
		sum_20 = 0.5 * count_20
		sum_21 = 0.5 * count_21
		sum_22 = 0.5 * count_22
		sum_23 = 0.5 * count_23
		#This sums up the weights and total count of occurences in the paragraph.
		sum_total = sum_1 + sum_2 + sum_3 + sum_4 + sum_5 + sum_6 + sum_7 + sum_8 + sum_9 + sum_10 + sum_11 + sum_12 + sum_13 + sum_14 + sum_15 + sum_16 + sum_17 + sum_18 + sum_19 + sum_20 + sum_22 + sum_23
		count_total = count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8 + count_9 + count_10 + count_11 + count_12 + count_13 + count_14 + count_15 + count_16 + count_17 + count_18 +  count_19 + count_20 + count_21 + count_22 + count_23
		#This filters out single-question paragraphs, which add noise to the summary and are usually redundant.
		if len(re.findall('\.', i)) < 1: #Single-question pars won't contain a period. Valuable single-sentence pars will contain a period. 
			average_score = 0
		#This filters out short paragraphs, which are likely to not contain any useful context. 
		#This doesn't apply when a speaker makes a correction.
		#Example taken from in 19405.pdf:
		#Senator Birmingham: The government has provided $2.1 million in the 2017-18 budget to support the authority. 
		#Funding arrangements for 2018-19 onwards will be subject to the usual budget processes.
		#Ms Evans: I might make a minor correction. There is a $1.456 million allocation. 
		#The briefing that we provided to the minister was incorrect.
		#Another example of extracted text:
		#Senator URQUHART: Again, this is to the minister, but if you want to flick it then I am sure you will. Can
		#you confirm that the allocation of money apportioned to the environment portfolio from the $1.1 billion in the
		#rollout of the National Landcare Programme moneys referred to in the budget will stay the same as previously?
		#Will it rise or will it fall between now and 2023?
		#Senator Birmingham: Just for the record, Senator Urquhart, it is $1.1 billion, not $1.1 million.
		elif len(i) <= 55: 
			if re.findall('incorrect', i) or re.findall('correction', i) or re.findall('mistake', i) or re.findall('should have said', i) or re.findall('I meant', i) or re.findall('for the record', i) or re.findall('For the record', i):
				average_score = 1 
			else:
				average_score = 0
		elif re.findall('incorrect', i) or re.findall('correction', i) or re.findall('mistake', i) or re.findall('should have said', i) or re.findall('I meant', i) or re.findall('for the record', i) or re.findall('For the record', i):
				average_score = 1 #Correcting a statement clould require an explaination of more than 55 chars. I'm working on 'sorry' in the context of a correction.
		elif sum_total == 0.0:
			average_score = 0 #Takes care of float division
		#This calculates the mean score of the paragraph. Simple. 
		else: 
			average_score = sum_total/count_total
		#The below threshold ensures that only paragraphs with a high enough score will be included in the summary.
		#Paragraphs that contain high weigthed content will always make it through to the summary for one or more occurences (average score always above threshold).
		#Paragraphs that contain med to low weighted content will need a combination of weights to make it through to the summary.
		#0.6 is not too low where summary includes a lot of noise and is not too high where summary misses a lot of important info.
		if average_score > 0.6:
			summary.append(i)
	return "\n\n".join(summary)
	
def function_2(text):
	paragraphs = text.split('\n\n')
	count_vect = CountVectorizer()
	bow_matrix = count_vect.fit_transform(paragraphs)
	normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
	similarity_graph = normalized_matrix * normalized_matrix.T #term frequency/inverse doc frequency applied
	similarity_graph.toarray()
	nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
	scores = nx.pagerank(nx_graph) #TextRank applied
	ranked = sorted(((scores[i],s) for i,s in enumerate(paragraphs)), reverse=True) #Sorts all paragraphs from highest to lowest scores
	ten_percent = int(round(10.00/100.00 * len(ranked)))
	ten_percent_high_scores = ranked[0:ten_percent]
	summary = [x[1] for x in ten_percent_high_scores] #Takes top 10%, so the paragraphs with the highest scores (does not disturb the rank order)
	return "\n\n".join(summary)

#Text taken from the user's uploaded PDF or URL, cleaned and formatted.
title_main = '''

'''

#Text taken from the user's uploaded PDF or URL, cleaned and formatted.
#This only includes the text from the body (footnotes, contents, etc can be automatically filtered out).
content= '''

'''

text = nltk.word_tokenize(content)

people_titles = ['Senator', 'Sen', 'Sen.', 'Mr', 'Mr.', 'Mrs', 'Mrs.', 'Miss', 'Miss.', 'Ms', 'Ms.', 'Dr', 'Dr.', 'Doctor', 'Prof', 'Prof.', 'Professor', 'Lady', 'Lord', 'Sir', 'Capt', 'Capt.', 'Captain', 'Major', 'The Hon', 'The Hon.', 'The Honerable', 'the Hon', 'the Hon.', 'the Honerable', 'Judge', 'Chair', 'Chief']
titles_uppercase = [title.upper() for title in people_titles]
for i in titles_uppercase:
	people_titles.append(i)
#print people_titles

people_names = []
CAPS_names_before_colon = re.findall('\w+\s+\w+(?=\s*:[^/])', content)
#print CAPS_names_before_colon
for i in CAPS_names_before_colon:
	nltk_undetected_names = nltk.word_tokenize(i)
	for j in nltk_undetected_names:
		people_names.append(j)
#print people_names
for i in text:
	target_words = ['friend', 'boyfriend', 'girlfriend', 'mother', 'father', 'husband', 'wife', 'fiance', 'parter', 'son', 'daughter', 'auntie', 'uncle', 'cousin', 'second cousin']
	for j in target_words:
		if i == j:
			following_word = text[text.index(j) + 1]
			if following_word[0].isupper() == True:
				people_names.append(following_word)
#print people_names
for i in text:
	for j in people_titles:
		if i == j:
			next_word = text[text.index(j) + 1]
			if next_word[0].isupper() == True:
				people_names.append(next_word)
				#print next_word
#print people_names
pos_tags = nltk.pos_tag(text)
#print pos_tags
person_tags = nltk.ne_chunk(pos_tags, binary = False)
for subtree in person_tags.subtrees():
	if subtree.label() == 'PERSON':
		leaves = subtree.leaves()
		for leaf in leaves:
			people_names.append(leaf[0])
			#deal with duplicates
#print people_names

selected_stopwords = ['a', 'an', 'am', 'and', 'are', "aren't", 'as', 'at', 'as', 'be', 'for', 'is', 'it', "it's", 'of', 'or', 'so', 'the', 'to']
stopwords_capitalized = [stopword[:1].upper() + stopword[1:] for stopword in selected_stopwords]
for i in stopwords_capitalized:
	selected_stopwords.append(i)
#print selected_stopwords 

selected_punctuation = ['.', ',', ':', '?']#NER performs better when incl punctuation, and prob depends on title, too. Remvove punct and titles after.

content_stripped_people_names = [word for word in text if word not in people_names]
content_stripped_titles = [word for word in content_stripped_people_names if word not in people_titles]
#content_stripped = content_stripped_titles #Even when using stopwords and punct, ne_chunk treats both 'the book' and 'book our tickets' as NN
#Write rules that if previous word the, a , his, her, their, etc, treat as noun, then delete punct scores and re-calc.
content_stripped_stopwords = [word for word in content_stripped_titles if word not in selected_stopwords]
content_stripped = [word for word in content_stripped_stopwords if word not in selected_punctuation]
#print content_stripped #Too many WHISH-WILSON hyphen names not being filtered out, possibly causing NN to be higher than JJ, as JJ was higher last time.

tags = nltk.pos_tag(content_stripped)
#print tags
counts = Counter(tag for word, tag in tags)
#print counts

#Normalise the counts (the proportion  of each)
total = sum(counts.values())
pos_proportion = dict((word, float(count)/total) for word, count in counts.items())
#print pos_proportion
pos_proportion_sorted = sorted(pos_proportion, key=pos_proportion.get, reverse=True)
#print pos_proportion_sorted

doc_features = {}
doc_features['PRP_top_POS_tag'] = []
doc_features['JJ_top_POS_tag'] = [] 
doc_features['dollar_proportion'] = [] 
doc_features['percentage_proportion'] = [] 

PRP_tag = []
for i in pos_proportion_sorted[0:3]:
	if i == 'PRP':
		PRP_tag.append(i)

if len(PRP_tag) == 0:
	doc_features['PRP_top_POS_tag'].append('false')
else:
	doc_features['PRP_top_POS_tag'].append('true')
	
JJ_tag = []
for i in pos_proportion_sorted[0:3]:
	if i == 'JJ':
		JJ_tag.append(i)

if len(JJ_tag) == 0:
	doc_features['JJ_top_POS_tag'].append('false')
else:
	doc_features['JJ_top_POS_tag'].append('true')
	
#print pos_proportion_sorted[0:3]
#print PRP_tag
#print JJ_tag

paragraphs = re.split('\n\n',content)

dollar_counts = float(len(re.findall('\$', content)))
paragraphs_count = float(len(paragraphs))
dollars_proportion = float(dollar_counts/paragraphs_count * 1000)
#print dollar_counts
#print paragraphs_count
#print dollars_proportion
doc_features['dollar_proportion'].append(dollars_proportion)

percentage_counts = float(len(re.findall('%|per cent|percent', content)))
paragraphs_count = float(len(paragraphs))
percentage_proportion = float(percentage_counts/paragraphs_count * 1000)
#print percentage_counts
#print paragraphs_count
#print percentage_proportion
doc_features['percentage_proportion'].append(percentage_proportion)
#print doc_features

document = []
if doc_features.get('JJ_top_POS_tag') == ['true'] and doc_features.get('dollar_proportion') > [35.00]:
	doc = 'facts and figures'
	document.append(doc)
elif doc_features.get('JJ_top_POS_tag') == ['true'] and doc_features.get('percentage_proportion') > [9.00] and doc_features.get('dollar_proportion') > [15.00]:
	doc = 'facts and figures'
	document.append(doc) 
elif doc_features.get('PRP_top_POS_tag') == ['true'] and doc_features.get('dollar_proportion') < [20.00]:
	doc = 'discussion'
	document.append(doc)
elif doc_features.get('JJ_top_POS_tag') == ['false'] and doc_features.get('dollar_proportion') > [40.00]:
	doc = 'facts and figures'
	document.append(doc)
elif doc_features.get('PRP_top_POS_tag') == ['false'] and doc_features.get('dollar_proportion') < [20.00]:
	doc = 'discussion'
	document.append(doc)
elif doc_features.get('JJ_top_POS_tag') == ['false'] and doc_features.get('dollar_proportion') < [35.00] and doc_features.get('percentage_proportion') > [30.00]:
	doc = 'facts and figures'
	document.append(doc)
else:
	doc = 'discussion'
	document.append(doc)
#print document

if document[0] == 'facts and figures':
	papyrus = function_1(content)
else:
	papyrus = function_2(content)

print title_main
print papyrus
