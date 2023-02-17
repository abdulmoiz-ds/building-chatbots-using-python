$ pip install virtualenv
$ virtualenv chatbot_env
$ source chatbot_env/bin/activate
$ pip install nltk scikit-learn flask
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text
pip install chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
chatbot = ChatBot('MyBot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.english')
while True:
    user_input = input("You: ")
    bot_response = chatbot.get_response(user_input)
    print("Bot: ", bot_response)
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence, words):
    sentence_words = preprocess(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)
import random

def train(X, y, model, optimizer, criterion, epochs=1000):
    for epoch in range(epochs):
        for i in range(len(X)):
            model.zero_grad()

            inputs = torch.from_numpy(X[i])
            targets = torch.from_numpy(y[i])

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
def predict(model, sentence, words, labels):
    model.eval()

    with torch.no_grad():
        input = bag_of_words(sentence, words)
        input = torch.from_numpy(input)

        output = model(input)
        output = output.detach().numpy()

        index = np.argmax(output)
        label = labels[index]
        return label
