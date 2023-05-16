import json
import spacy
from spacy.matcher import PhraseMatcher

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# Load the skill extractor
skill_extractor = pipeline.load('skill_extractor')

# Define a function that takes in a text and returns a list of skills
def extract_skills(text):
    try:
        # Get the annotations for the text
        annotations = skill_extractor.annotate(text)
        matches = []
        for skill in annotations['results']['full_matches']:
            # Add the skill to the list
            matches.append(skill['doc_node_value'])
        return matches
    except:
        return []

filenames = ["us_profiles.txt","canada_profiles.txt","india_profiles.txt","singapore_profiles.txt"]
all_profiles = []
count = 1
for filename in filenames:
    with open(filename, "r") as file:
        # loop through each profile
        for line in file:
            try:
                print("Processing item number "+str(count), end="\r")
                dict_obj = json.loads(line.strip())
                if dict_obj['summary'] == None:
                    dict_obj['summary'] = ""
                summary = dict_obj['summary']
                # loop through each experience of the profile
                for exp in dict_obj['experiences']:
                    if 'summary' in exp.keys() and exp['summary'] != None:
                        summary += exp['summary']
                        summary += " "
                    if 'description' in exp.keys() and exp['description'] != None:
                        summary += exp['description']
                        summary += " "

                if 'skills' not in dict_obj.keys():
                    dict_obj['skills'] = []

                # extract skills/keywords from summary
                if dict_obj['skills'] == [] or dict_obj['skills'] == None: # and dict_obj['skills'] != None:
                    dict_obj['skills'] = [*set(extract_skills(summary))]
                else:
                    dict_obj['skills'] += [*set(extract_skills(summary))]
                    dict_obj['skills'] = [*set(dict_obj['skills'])]
                
                # append profile to list only if skills are present
                if dict_obj['skills'] != []:
                    all_profiles.append(dict_obj)

                count += 1
            except:
                pass

        

print(len(all_profiles))

filename = "all_profiles.json"

with open(filename, "w") as file:
    json.dump(all_profiles, file)


