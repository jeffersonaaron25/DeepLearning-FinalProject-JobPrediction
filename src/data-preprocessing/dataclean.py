import re
import numpy as np
import pandas as pd
import json
import torch
import spacy 
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from re import sub
from gensim.utils import simple_preprocess

import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity

# Load the model: this is a big file, can take a while to download and open
glove = api.load("glove-wiki-gigaword-50")    
similarity_index = WordEmbeddingSimilarityIndex(glove)

stopwords = ['the', 'and', 'are', 'a']

# This code is used to normalize job titles using similarity to match with the list

normalized_titles = [
'Account Executive',
'Account Manager',
'Accountant',
'Administration Manager',
'Administrative Assistant',
'Advertising/Public Relations Professional',
'Advocate/Solicitor',
'AI Specialist',
'Air Cargo Officer',
'Air Transport Service Supervisor',
'Application Developer',
'Architect (Building)',
'Artist or Designer',
'Astronomer',
'Attorney',
'Auditor',
'Automotive Engineer',
'Backend Developer',
'Biologist',
'Branding',
'Business Analyst',
'Business Consultant',
'Business Development',
'Business Development Executive/Manager',
'Buyer/Supply Chain/Procurement Manager',
'Chief Executive Officer',
'Chief Financial Officer',
'Chemical Engineer',
'Chemist',
'Chief Operations Officer',
'Civil Engineer',
'Compliance Manager',
'Construction Manager',
'Content Marketing',
'Content Writer',
'Copywriter',
'Customer Success Manager',
'Data Analyst',
'Data Engineer',
'Data Scientist',
'DevOps Engineer',
'Digital Marketing',
'Doctor',
'Electrician',
'Electrical Engineer',
'Employee Relations Specialist',
'Engineering Manager/Project Manager',
'Executive Assistant',
'Executive Director',
'Film Crew',
'Financial Advisor',
'Financial Analyst',
'Financial Planner',
'Frontend Developer',
'Full Stack Developer',
'Game Developer',
'Graphic Designer',
'Investment Banker',
'Investment Specialist',
'IT Consultant',
'IT Project Manager',
'Journalist',
'Lawyer',
'Legal Assistant',
'Legal Counsel',
'Logistics Manager',
'Management Consultant',
'Manufacturing Engineer',
'Market Analyst',
'Marketing Consultant',
'Marketing Executive/Manager',
'Materials Engineer',
'Mechanic',
'Mechanical Engineer',
'Medical Assistant',
'Medical Laboratory Scientist',
'Medical Resident',
'Network Engineer',
'Nurse',
'Nutritionist',
'Office Manager',
'Operations Manager',
'Paralegal',
'Pharmacist',
'Photographer',
'Physician',
'Physicist',
'Physiotherapist',
'Planning Manager',
'Product Manager',
'Production Assistant',
'Production Engineer',
'Professor/Lecturer',
'Program Manager',
'Project Engineer',
'Project Manager',
'Psychiatrist',
'Psychologist',
'Quality Control/Assurance Manager',
'Quality Engineer',
'Quality Manager',
'Recruitment Specialist',
'Research Analyst',
'Research Assistant',
'Research Associate',
'Research Scientist',
'Restaurant Manager/Supervisor',
'Retail Manager',
'Sales Associate',
'Sales Manager',
'Sales Representative',
'Scientist',
'Security Engineer',
'Security Manager',
"Service Engineer/Manager",
"Social Work Associate",
"Software Engineer/Developer",
"Speech Therapist",
"Surveyor",
"Surgeon",
"Systems Analyst/Consultant",
"Telecommunications Engineer",
"UI/UX Designer",
"Veterinarian",
"Web Developer"
]

# From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
def preprocess(doc):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

corpus = [preprocess(document) for document in normalized_titles]

def json_to_data(json_file_path):
    data = None
    with open(json_file_path) as f:
        data = json.load(f)
    return data

json_files = ['sg_profiles.json', "us_profiles.json", "in_profiles.json", "ca_profiles.json"]
job_titles = {}
tfidf_vectorizer = TfidfVectorizer()
count = 1
for file_name in json_files:
    data = json_to_data(file_name)
    for item in data:
        try:
            for exp in item['experiences']:
                exp['normalized_job_title'] = None
                if len(exp['title'].split()) < 2:
                    continue
                if 'student' in exp['title'].lower().split():
                    continue
                if exp['title'].lower() not in job_titles:   
                    # Preprocess the documents, including the query string
                    query = preprocess(exp['title'].lower())

                    # Build the term dictionary, TF-idf model
                    dictionary = Dictionary(corpus+[query])
                    tfidf = TfidfModel(dictionary=dictionary)

                    # Create the term similarity matrix.  
                    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

                    query_tf = tfidf[dictionary.doc2bow(query)]

                    index = SoftCosineSimilarity(
                                tfidf[[dictionary.doc2bow(document) for document in corpus]],
                                similarity_matrix)

                    doc_similarity_scores = index[query_tf]

                    # Output the sorted similarity scores and documents
                    sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
                    
                    for idx in sorted_indexes:
                        if doc_similarity_scores[idx] > 0.8:
                            print(count,exp['title'].lower())
                            print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {normalized_titles[idx]}')
                            if normalized_titles[idx].lower() not in job_titles.keys():
                                job_titles[normalized_titles[idx].lower()] = 1
                            else:
                                job_titles[normalized_titles[idx].lower()] += 1
                            count += 1
                            exp['normalized_job_title'] = normalized_titles[idx]
                            
                        break
                else:
                    job_titles[exp['title'].lower()] += 1
                    exp['normalized_job_title'] = exp['title'].lower()
                count += 1
        except:
            pass
    with open(file_name.split(".")[0]+"_normalized.json", "w") as file:
        json.dump(data, file)

print(len(job_titles))
print(job_titles)
                

