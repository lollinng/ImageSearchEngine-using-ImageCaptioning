import re
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
nltk.download('punkt')


stop_words = set(stopwords.words('english'))


def Doct2vect(documents):
    """
    used to create embeddings and find simmilarities between the first index of list and other indexes of the list.
    : param documnets : a list for calculating the consine similarity of first element with others

    reference : https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630
    """

    documents_df=pd.DataFrame(documents,columns=['documents'])
    documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words))
    tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(documents_df.documents_cleaned)]
    model_d2v = Doc2Vec(vector_size=100,alpha=0.025, min_count=1)
    
    model_d2v.build_vocab(tagged_data)

    for epoch in range(100):
        model_d2v.train(tagged_data,
                    total_examples=model_d2v.corpus_count,
                    epochs=model_d2v.epochs)
        
    document_embeddings=np.zeros((documents_df.shape[0],100))

    for i in range(len(document_embeddings)):
        document_embeddings[i]=model_d2v.dv[i]
        
    pairwise_similarities= np.array(cosine_similarity(document_embeddings)[0])
    normalizedData = (pairwise_similarities-np.min(pairwise_similarities))/(np.max(pairwise_similarities)-np.min(pairwise_similarities))
    return normalizedData

    


if __name__ == "__main__":
    #  # Sample corpus
    documents = ['Machine learning is the study of computer algorithms that improve automatically through experience.\
    Machine learning algorithms build a mathematical model based on sample data, known as training data.\
    The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
    where no fully satisfactory algorithm is available.',
    'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
    The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
    'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
    It involves computers learning from data provided so that they carry out certain tasks.',
    'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal"\
    or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
    'Software engineering is the systematic application of engineering approaches to the development of software.\
    Software engineering is a computing discipline.',
    'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
    about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
    Developing a machine learning application is more iterative and explorative process than software engineering.'
    ]
    
    print(Doct2vect(documents))
    