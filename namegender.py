def gender_features(word):
    return {'last_three' : word[-3:], 'first four' : word[:4]}

from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk import classify
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])

import random
random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = NaiveBayesClassifier.train(train_set)


print("Wyatt is..."+classifier.classify(gender_features('Wyatt')))

print(classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(30)