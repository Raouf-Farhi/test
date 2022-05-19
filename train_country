import spacy
from spacy.lang.fr import French
from spacy.pipeline import EntityRuler
from spacy.language import Language
import json
from spacypdfreader import pdf_reader


#we create a function to load our json file :
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return (data)


#function to generate countries :
def generate_countries(file):
    data_countries = load_data(file)
    new_countries = []
    for country in data_countries:
        new_countries.append(country)
    #clean our data()
    for country in data_countries:
        country = country.replace("le", "").replace("Le", "").replace("La", "").replace("la", "").replace("et", "").replace("Et", "")
        #split all the names :
        names = country.split()
        for name in names :
            name = name.strip()
            new_countries.append(name)
        #get rid of the parethesis:
        if "(" in country:
            names = country.split("(")
            for name in names :
                name = name.replace(")", "").strip()
                new_countries.append(name)
                #print(name)
                #found that we neeed to get rid of spaces :
                if " " in name:
                    new_names = name.split()
                    for x in new_names :
                        new_countries.append(x)
                new_countries.append(name)

#to get rid of duplicates : convert to a set (a list without ducplicates) and than to a list againe
    final_countries = list(set(new_countries))
    return (final_countries)

#Now we are gonna create our training data :
def training_data(file, type):
    patterns = []
    data = generate_countries(file) #we pass our file of training to our function
    for item in data :
        pattern = {
                    "label": type, #the type of label that we are going to pass (in our case country)
                    "pattern": item

        }
        #in order to train spacy we need to create patterns for it, a dictionnary that consists of atleast (a label and a pattern)
        patterns.append(pattern)
    return patterns
#we created a long dictionay that contains paterns for individual with the label "COUNTRY"


#Next step is creating rules :
@Language.component("component")
def generate_rules(patterns):
    nlp = spacy.load("fr_core_news_lg")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    #we passed the whole list of patterns to our EntityRuler
    #we create a spacy pipeline with our EntityRuler () and we save it
    nlp.to_disk("country_ner")

patterns = training_data(r"C:\Users\MasterPro\Documents\rev\paysfr.json", "COUNTRY")
#once we generate our model we comment it because we don't need to generate it anymore :

#generate_rules(patterns)

#NOW WE TEST OUR MODEL :
nlp = spacy.load("country_ner")
def test_model(model, text):
    doc = nlp(text)  #takes our text and run our model over it
    results = []
    for ent in doc.ents:
        results.append(ent.text)
    return results


text = pdf_reader(r'C:\Users\MasterPro\Documents\rev\pdf1.pdf', nlp)
list_country = []
results = test_model(nlp, text)
for result in results:
     list_country.append(result)



print(list_country)
