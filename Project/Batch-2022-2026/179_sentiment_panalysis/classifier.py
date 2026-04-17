from threading import Lock
import joblib
import re
from itertools import islice
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


COMMENTS_VOLUME = 1500

LABEL_NAMES = {
    0: 'Positive',
    1: 'Negative',
    2: 'Neutral',
    3: 'Spam',
    4: 'Interrogative',
    5: 'Corrective',
    6: 'Imperative',
}


class ClassiferSingleton(metaclass=SingletonMeta):
    comment_scraper = YoutubeCommentDownloader()
    model = None
    vectorizer = None

    def set_paths(self, model_path: str, vectorizer_path: str) -> None:
        self.model = joblib.load(filename=model_path)
        self.vectorizer = joblib.load(filename=vectorizer_path)

    def make_analysis(self, video_url):
        comments_generator = self.comment_scraper.get_comments_from_url(
            youtube_url=video_url, sort_by=SORT_BY_POPULAR
        )

        dirty_comments = [
            comment['text']
            for comment in islice(comments_generator, COMMENTS_VOLUME)
        ]

        if not dirty_comments:
            print("Warning: No comments were fetched.")
            return []

        clean_comments = self._clean(dirty_comments)
        features = self.vectorizer.transform(clean_comments).toarray()
        predictions = self.model.predict(features)
        counter = collections.Counter(predictions)

        labels = [LABEL_NAMES.get(i, str(i)) for i in sorted(LABEL_NAMES.keys())]
        heights = [
            counter.get(i, 0) / len(predictions) * 100
            for i in sorted(LABEL_NAMES.keys())
        ]
        colors = ['green', 'red', 'yellow', 'gray', 'blue', 'orange', 'purple']

        os.makedirs('static/vis', exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.bar(x=labels, height=heights, color=colors)
        plt.title(f'Comments Sentiment (Volume: {len(predictions)} comments)')
        plt.xlabel('Category')
        plt.grid(axis='y', zorder=0)
        plt.ylim(0, 100)
        plt.ylabel('% of comments')
        plt.tight_layout()
        plt.savefig('static/vis/plot.png')
        plt.close()

        return predictions

    def _clean(self, comments):
        processed_features = []
        for comment in comments:
            processed_feature = re.sub(r'\W', ' ', str(comment))
            processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
            processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
            processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
            processed_feature = re.sub(r'^b\s+', '', processed_feature)
            processed_feature = processed_feature.lower()
            processed_features.append(processed_feature)
        return processed_features