from pathlib import Path
from datetime import datetime
import json

import tweepy

from data import TWITTER_DIR

with open(Path.expanduser(Path('~/.secrets/twitter_api_key')), 'r') as f:
    TWITTER_API_KEY = f.read().strip()
with open(Path.expanduser(Path('~/.secrets/twitter_secret_key')), 'r') as f:
    TWITTER_SECRET_KEY = f.read().strip()
with open(Path.expanduser(Path('~/.secrets/twitter_access_token')), 'r') as f:
    TWITTER_ACCESS_TOKEN = f.read().strip()
with open(Path.expanduser(Path('~/.secrets/twitter_secret_token')), 'r') as f:
    TWITTER_SECRET_TOKEN = f.read().strip()

# print(TWITTER_API_KEY, TWITTER_SECRET_KEY)

auth = tweepy.AppAuthHandler(TWITTER_API_KEY, TWITTER_SECRET_KEY)
api = tweepy.API(auth)


def _search_archive(query):
    return api.search_full_archive('dev', query)


def search_archive(query, save=True):
    tweets = _search_archive(query)
    tweets_json = list(map(lambda t: t._json, tweets))
    if save:
        ts_str = datetime.now().strftime(f'%Y%m%d_{query}_full_archive.json')
        with open(TWITTER_DIR.joinpath('tweets', ts_str), 'w+'):
            cur = json.load(f)
            # TODO: Check for duplicates
            json.dump(cur + tweets_json, f)
    return tweets


def print_tweets(tweets):
    for tweet in tweets:
        print('='*100)
        print(tweet._json['text'])
        print('='*100)
