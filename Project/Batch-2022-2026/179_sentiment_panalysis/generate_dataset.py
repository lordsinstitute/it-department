import pandas as pd
import random

random.seed(42)

positive = [
    "This is absolutely amazing, loved every second of it!",
    "Great video, very informative and well explained.",
    "Thank you so much for sharing this, really helpful!",
    "One of the best videos I have watched in a long time.",
    "Fantastic content, keep up the great work!",
    "This made my day, so inspiring and motivating.",
    "Excellent explanation, very clear and easy to follow.",
    "I learned so much from this, thank you!",
    "Beautifully made, the quality is outstanding.",
    "You are incredibly talented, this is top notch.",
    "So well done, I watched it three times already.",
    "This channel deserves way more subscribers.",
    "Perfect video, exactly what I was looking for.",
    "Really appreciate the effort you put into this.",
    "This is pure gold, thank you for the hard work.",
]

negative = [
    "This is terrible, complete waste of my time.",
    "Worst video I have seen, very disappointing.",
    "The audio quality is awful, could not understand anything.",
    "This is completely wrong and misleading information.",
    "I do not recommend this at all, very bad content.",
    "Really boring and poorly made, thumbs down.",
    "This channel is going downhill fast.",
    "So many errors in this video, very unprofessional.",
    "I regret watching this, total garbage.",
    "The editing is horrible and the pacing is off.",
    "Unsubscribed after watching this disaster.",
    "Nothing useful here, just filler content.",
    "This is the most confusing explanation I have ever seen.",
    "Clickbait title, the video delivers nothing.",
    "Very rude presenter, extremely off-putting.",
]

neutral = [
    "I watched this video.",
    "This is a video about the topic.",
    "The video was uploaded today.",
    "I found this in my recommendations.",
    "This video is about twelve minutes long.",
    "The presenter talks about several points.",
    "There are subtitles available for this video.",
    "I came across this while browsing.",
    "The video has a few chapters.",
    "This was suggested to me by YouTube.",
    "I watched this on my phone.",
    "The video covers the basics.",
    "There are links in the description.",
    "The thumbnail caught my attention.",
    "I paused this halfway through.",
]

spam = [
    "Click here to win a free iPhone now!!!",
    "Subscribe to my channel for free money giveaway!",
    "Visit my website for amazing deals, link in bio!",
    "I make 500 dollars a day from home, ask me how!",
    "Free followers and likes at my profile, check it out!",
    "Buy cheap views and subscribers now, DM me!",
    "This is a scam avoid at all costs, visit my page instead!",
    "Promo code SPAM50 for discount on my useless product!",
    "First comment! Subscribe to me for no reason!",
    "Check my latest video for free robux and vbucks!",
    "I got 10000 subscribers in one day, here is how!",
    "Dating site for singles, click the link below!",
    "Make money online fast, no experience needed!",
    "Free crypto airdrop, send me your wallet address!",
    "Bot followers for sale, cheapest rates guaranteed!",
]

interrogative = [
    "Can someone explain what this means?",
    "What software did you use to make this video?",
    "Does anyone know where I can find more like this?",
    "Why does this happen the way it does?",
    "Is this method still relevant in the current year?",
    "How long did it take you to create this?",
    "What do you think about the alternative approach?",
    "Can you make a follow up video on this topic?",
    "Where can I download the files you used here?",
    "Has anyone tried this on a Mac instead of Windows?",
    "What is the name of the song in the background?",
    "Is this suitable for complete beginners?",
    "How many hours did you spend on this project?",
    "Why is my result different from what you showed?",
    "Do you have a written version of this tutorial?",
]

corrective = [
    "Actually that is not correct, the answer is different.",
    "You made a mistake at around the three minute mark.",
    "The formula you used there is actually wrong.",
    "I think you got the steps in the wrong order here.",
    "That is a common misconception, the real answer is different.",
    "Small error in the code, you forgot a closing bracket.",
    "The date you mentioned is actually off by one year.",
    "You mislabeled the diagram in this section.",
    "That definition is not quite accurate, please check again.",
    "The spelling of that term is incorrect in your slides.",
    "You switched the x and y axes in that chart.",
    "I believe you meant to say the opposite of what you said.",
    "That calculation gives a different result when done correctly.",
    "The link you shared in the description is broken.",
    "You confused two different concepts in this part.",
]

imperative = [
    "Please make more videos on this subject!",
    "Do not forget to add subtitles next time.",
    "Pin this comment so everyone can see it.",
    "Add timestamps to your videos please.",
    "Stop using that background music, it is distracting.",
    "Please respond to your comments more often.",
    "Make a part two of this video immediately!",
    "Turn on the captions for accessibility.",
    "Share this video with everyone you know.",
    "Do not skip the introduction next time please.",
    "Please slow down when explaining complex parts.",
    "Upload more frequently, we need this content!",
    "Fix the audio levels in your next upload.",
    "Please include the source links in the description.",
    "Do not cut the video so abruptly at the end.",
]

categories = {
    "positive": (positive, 0),
    "negative": (negative, 1),
    "neutral": (neutral, 2),
    "spam": (neutral, 3),
    "interrogative": (interrogative, 4),
    "corrective": (corrective, 5),
    "imperative": (imperative, 6),
}

label_map = {
    0: "positive",
    1: "negative",
    2: "neutral",
    3: "spam",
    4: "interrogative",
    5: "corrective",
    6: "imperative",
}

all_sentences = []
all_labels = []

templates_per_category = {
    0: positive,
    1: negative,
    2: neutral,
    3: spam,
    4: interrogative,
    5: corrective,
    6: imperative,
}

for label_id, sentences in templates_per_category.items():
    for _ in range(7143):
        base = random.choice(sentences)
        words = base.split()
        if random.random() > 0.5 and len(words) > 4:
            idx = random.randint(0, len(words) - 1)
            words[idx] = words[idx] + "!"
        variation = " ".join(words)
        all_sentences.append(variation)
        all_labels.append(label_id)

combined = list(zip(all_sentences, all_labels))
random.shuffle(combined)
all_sentences, all_labels = zip(*combined)

df_50k = pd.DataFrame({
    "id": range(1, len(all_sentences) + 1),
    "text": all_sentences,
    "label": all_labels
})

df_50k = df_50k.iloc[:50000]
df_50k.to_csv("data/generic_sentiment_dataset_50k.csv", index=False)

df_10k = df_50k.iloc[:10000].copy()
df_10k.to_csv("data/generic_sentiment_dataset_10k.csv", index=False)

print(f"Datasets generated successfully!")
print(f"50k dataset: {len(df_50k)} rows")
print(f"10k dataset: {len(df_10k)} rows")
print(f"Label distribution:\n{df_50k['label'].value_counts()}")