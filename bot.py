import discord
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tflearn 
import tensorflow as tf 
import numpy as np

from dotenv import load_dotenv
load_dotenv()

import train
import os

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in')

@client.event
async def on_message(message):
    if message.content.startswith('/'):
        train.model.load("Model.tflearn")

        inp = message.content
        inp = list(inp)
        inp.pop(0)
        enter = ""
        for i in inp:
            enter += i
        
        results = train.model.predict([bag_of_words(enter, train.words)])
        results_index = np.argmax(results)
        tag = train.labels[results_index]
        
        for tg in train.data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        await message.channel.send(random.choice(responses))
    else:
        pass

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def message():
    while True:
        inp = bot.on_message.message.content
        inp = list(inp)
        inp.remove(0)
        enter = ""
        for i in inp:
            enter += i
        
        results = train.model.predict([bag_of_words(enter, train.words)])
        results_index = np.argmax(results)
        tag = train.labels[results_index]
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']


client.run(os.getenv("TOKEN"))
