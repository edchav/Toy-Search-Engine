import nltk
import os 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math


def lower_case():
    corpusroot = './US_Inaugural_Addresses'
    inaugural_addresses = {}
    for filename in os.listdir(corpusroot):
        if filename.endswith('.txt'): #filename.startswith('0') or filename.startswith('1') or filename.startswith('2') or filename.startswith('3'):
            file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
            doc = file.read()
            file.close() 
            doc = doc.lower()
            inaugural_addresses[filename] = doc
    return inaugural_addresses

def tokenize(inaugural_addresses):
    tokenizer = RegexpTokenizer(r'[a-zA-z]+')
    inaugural_tokens = {}
    for filename, text in inaugural_addresses.items():
        # tokenize the text
        inaugural_tokens[filename] = tokenizer.tokenize(text)
    return inaugural_tokens

def remove_stopwords(inaugural_tokens):
    english_stopwords = stopwords.words('english')
    inaugural_tokens_nostopwords = {}
    for filename, tokens in inaugural_tokens.items():
        inaugural_tokens_nostopwords[filename] = []
        for token in tokens:
            if token not in english_stopwords:
                inaugural_tokens_nostopwords[filename].append(token)
    return inaugural_tokens_nostopwords

def apply_stemmer_docs(inaugural_tokens_nostopwords):
    stemmer = PorterStemmer()
    inaugural_stemmed = {}
    for filename, tokens in inaugural_tokens_nostopwords.items():
        inaugural_stemmed[filename] = []
        for token in tokens:
            inaugural_stemmed[filename].append(stemmer.stem(token))
    return inaugural_stemmed

def apply_stemmer_token(token):
    stemmer = PorterStemmer()
    return stemmer.stem(token)

def get_raw_term_frequency(inaugural_stemmed):
    raw_term_frequency = {}
    for filename, tokens in inaugural_stemmed.items():
        raw_term_frequency[filename] = {}
        for token in tokens:
            if token in raw_term_frequency[filename]:
                raw_term_frequency[filename][token] +=1
            else:
                raw_term_frequency[filename][token] = 1
    return raw_term_frequency

def getidf(token):
    stemmed_token = apply_stemmer_token(token)
    total_docs = len(inaugural_stemmed)

    document_frequency = 0
    for tokens in inaugural_stemmed.values():
        if stemmed_token in tokens:
            document_frequency += 1
    if document_frequency == 0:
        return -1
    inverse_document_frequency = math.log10(total_docs / document_frequency)
    return inverse_document_frequency

def normalize_values(inaugural_stemmed, raw_term_frequency):
    normalized_values = {}
    for filename, tokens in inaugural_stemmed.items():
        magnitude = 0
        for token in set(tokens):
            inverse_document_frequency = getidf(token)
            term_frequency = raw_term_frequency[filename].get(token, 0)
            term_frequency_inverse_document_frequency = (1 + math.log10(term_frequency)) * inverse_document_frequency
            magnitude += term_frequency_inverse_document_frequency ** 2
        normalized_values[filename] = math.sqrt(magnitude)
    return normalized_values

def getweight(filename, token):
    stemmed_token = apply_stemmer_token(token)
    term_frequency = raw_term_frequency[filename].get(stemmed_token, 0)
    if term_frequency == 0:
        return 0
    inverse_document_frequency = getidf(stemmed_token)
    term_frequency_inverse_document_frequency = (1 + math.log10(term_frequency)) * inverse_document_frequency
    return term_frequency_inverse_document_frequency / normalized_values[filename]

def query(qstring):
    tokenizer = RegexpTokenizer(r'[a-zA-z]+')
    query_tokens = tokenizer.tokenize(qstring.lower())
    english_stopwords = stopwords.words('english')
    query_stemmed = [apply_stemmer_token(token) for token in query_tokens if token not in english_stopwords]
    
    query_tf = {token: 0 for token in query_stemmed}
    for token in query_stemmed:
        if token in query_tf:
            query_tf[token] += 1

    query_log_tf = {}
    for token, term_frequency in query_tf.items():
        if term_frequency > 0:
            query_log_tf[token] = 1 + math.log10(term_frequency)
    
    query_magnitude = 0
    for tf_idf in query_log_tf.values():
        query_magnitude += tf_idf ** 2
    query_magnitude = math.sqrt(query_magnitude)

    query_normalized = {}
    for token, tf_idf in query_log_tf.items():
        query_normalized[token] = tf_idf / query_magnitude

    doc_score = {}
    for filename, tokens in inaugural_stemmed.items():
        score = 0
        for token in query_normalized:
            if token in tokens:
                query_weight = query_normalized[token]
                doc_weight = getweight(filename, token)
                score += query_weight * doc_weight
        doc_score[filename] = score

    best_doc = max(doc_score, key=doc_score.get)
    best_score = doc_score[best_doc]
    return (best_doc, best_score)


inaugural_addresses = lower_case()
#print(inaugural_addresses.keys())
#print("list of addresses", inaugural_addresses['01_washington_1789.txt'][:1000])

inaugural_tokens = tokenize(inaugural_addresses)
#print("inaugrual tokens: ", inaugural_tokens['01_washington_1789.txt'][:10])

inaugural_tokens_nostopwords = remove_stopwords(inaugural_tokens)
#print("without stopwords", inaugural_tokens_nostopwords['01_washington_1789.txt'][:10])

inaugural_stemmed = apply_stemmer_docs(inaugural_tokens_nostopwords)
#print("lenght of stemmed", len(inaugural_stemmed))
#print("stemmed", inaugural_stemmed['01_washington_1789.txt'][:10])

raw_term_frequency = get_raw_term_frequency(inaugural_stemmed)
#print("raw term frequency", raw_term_frequency['01_washington_1789.txt'])

normalized_values = normalize_values(inaugural_stemmed, raw_term_frequency)


print("%.12f" % getidf('children'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('people'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")

print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('05_jefferson_1805.txt','press'))
print("--------------")

print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("cuba government"))
