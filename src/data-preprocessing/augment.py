import random
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('jobsdata_refined.csv', encoding='utf-8')

# iterate through the df and remove duplicates of job titles
keywords_list = {}
for i, row in tqdm(df.iterrows()):
    job_title = row['job_title']
    keywords = row['keywords']
    if keywords not in keywords_list.keys():
        keywords_list[keywords] = [job_title]
    else:
        if job_title not in keywords_list[keywords]:
            keywords_list[keywords].append(job_title)

df = pd.DataFrame(columns=['job_title','keywords'])
j = 0 
for keywords in tqdm(keywords_list.keys()):
    for job_title in keywords_list[keywords]:
        df.loc[j, ['job_title']] = job_title
        df.loc[j, ['keywords']] = keywords
        j += 1


# The code below is for data augmentation
j = df.index[-1] + 1

job_map = {}
job_count = {}
for i, row in tqdm(df.iterrows()):
    job_title = row['job_title']
    keywords = row['keywords'].split(', ')
    
    if job_title not in job_map:
        job_map[job_title] = set()
        job_count[job_title] = 0
    
    job_map[job_title].update(keywords)
    job_count[job_title] += 1

for job_title in job_map.keys():
    job_map[job_title] = list(job_map[job_title])

for k in range(10):
    for job_title in tqdm(job_count.keys()):
        for i in range(job_count[job_title]):
            kidx = random.randint(5, 10)
            keywords = ""
            for idx in range(kidx):
                if idx != 0:
                    keywords += ", "
                keyid = random.randint(0, len(job_map[job_title])-1)
                keywords += job_map[job_title][keyid]

            df.loc[j, ['job_title']] = job_title
            df.loc[j, ['keywords']] = keywords
            j += 1

df.to_csv('jobsdata_augmented_large.csv', encoding='utf-8', index=False)