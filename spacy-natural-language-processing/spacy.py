import spacy
from spacy import displacy
from pathlib import Path

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

# Process whole documents
text = (u"After a long night at the Crucifix, we all left to go to the "
        u"basement of my parents house. What we found there, was incredible. "
        u"An American fellow, who was able to control frequencies of light."
        u"We tried to catch him and put him into the Box of Descrution, but we failed.")
doc = nlp(text)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

# displacy.serve(doc, style="dep")
html = displacy.render(doc, style='dep', page=True)
svg = displacy.render(doc, style='dep')
output_path = Path('sentence.svg')
output_path.open('w', encoding='utf-8').write(svg)
# Determine semantic similarities
doc1 = nlp(u"my fries were super gross")
doc2 = nlp(u"such disgusting fries")
similarity = doc1.similarity(doc2)
print(doc1.text, doc2.text, similarity)
