import re
import nltk
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize



# Download the necessary NLTK resources
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")

# Set and save the english stopwords
stop_words = set(stopwords.words("english"))



# Specify whether each title is a question sentence
def is_question(title):
    title = str(title).strip().lower()
    
    question_words = ["who", "what", "why", "how", "is", "are", "can", "should", "do", "did", "does", "could", "would", "will"]
    first_word = re.split(r'\W+', title)[0] 
    
    if (title.endswith("?")) or (first_word in question_words):
        return 1
    
    return 0

# Specify the number of capital words in each title
def count_capital_words(title):
    words = str(title).split()

    return sum(1 for word in words if word.isupper() and len(word) > 1)

# Specify the number of emotional words in each title
def count_emotional_words(title):
    words = str(title).lower().split()

    emotional_words = {
    "amazing", "awesome", "fantastic", "miracle", "incredible", "wonderful", "blessed", "enjoy", "love", "loved", "loving",
    "success", "successful", "best", "win", "winning", "victory", "brilliant", "joy", "joyful", "celebrate", "celebration",
    "great", "beautiful", "inspiring", "hope", "hopeful", "charming", "graceful", "positive", "hero", "heroic", "legend", "legendary",
    "breakthrough", "remarkable", "heartwarming", "uplifting", "grateful", "gratitude", "transformative", "unstoppable",
    "terrible", "horrible", "disaster", "tragic", "tragedy", "worst", "evil", "corrupt", "corruption", "sad", "saddening",
    "pathetic", "dirty", "shameful", "angry", "furious", "disgusting", "gross", "outrage", "disgrace", "abuse", "abusive",
    "cruel", "painful", "hateful", "hate", "vile", "brutal", "toxic", "oppressive", "heinous", "wicked", "ruined", "wrecked",
    "fear", "panic", "scary", "deadly", "dangerous", "crisis", "urgent", "warning", "threat", "explosive", "devastating",
    "emergency", "risk", "catastrophe", "fatal", "collapse", "chaos", "attack", "terror", "terrorist", "massacre",
    "biohazard", "mutation", "war", "hijack", "carnage", "plague", "outbreak", "lockdown", "invasion", "blackout",
    "unbelievable", "shocking", "bizarre", "outrageous", "unexpected", "secret", "exposed", "leaked", "exclusive", "hidden", "revealed",
    "mystery", "cover-up", "controversial", "chaotic", "eye-opening", "jaw-dropping", "bombshell", "truth", "insane", 
    "insanity", "unthinkable", "censored", "forbidden", "mind-blowing", "disturbing", "dark", "underground", "controversy"
}


    return sum(1 for word in words if word.strip(".,!?\"'") in emotional_words)

# Specify the number of stopwords in each text
def count_stopwords(text):
    words = str(text).lower().split()

    return sum(1 for word in words if word in stop_words)

# Specify the ratio of unique words to total words in each text
def lexical_diversity(text):
    words = str(text).lower().split()

    if len(words) == 0:
        return 0
    
    return len(set(words)) / len(words)



# Create the directory structure for the datasets file
base_path_read = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
read_fake_file = os.path.join(base_path_read, "Fake.csv")
read_true_file = os.path.join(base_path_read, "True.csv")

# Load the fake and true news datasets into a pandas dataframe
df_fake = pd.read_csv(read_fake_file, rows=500)
df_true = pd.read_csv(read_true_file, rows=500)

# Add a label fake/true for each news
df_fake["label"] = "fake"
df_true["label"] = "true"

# Merge and join the two dataframes into a single dataframe
df_all = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# Delete rows containing any NaN value
df_all = df_all.dropna(subset=["title", "text", "subject", "date"])

# Convert the date column to datetime format
df_all["date"] = df_all["date"].str.strip().str.lower()
df_all["date"] = pd.to_datetime(df_all["date"], format="%B %d, %Y", errors="coerce")


# Extracting the features from previous functions and columns


df_all["title_capital_word_count"] = df_all["title"].apply(count_capital_words)

df_all["is_question_title"] = df_all["title"].apply(is_question)

df_all["title_emotional_word_count"] = df_all["title"].apply(count_emotional_words)

df_all["title_word_count"] = df_all["title"].apply(lambda x: len(str(x).split()))

df_all["text_word_count"] = df_all["text"].apply(lambda x: len(str(x).split()))

df_all["text_stopword_count"] = df_all["text"].apply(count_stopwords)

df_all["text_stopword_ratio"] = df_all["text_stopword_count"] / (df_all["text_word_count"] + 1e-5)

df_all["text_sentence_count"] = df_all["text"].apply(lambda x: len(sent_tokenize(str(x))))

df_all["text_lexical_diversity"] = df_all["text"].apply(lexical_diversity)

df_all["title_number_count"] = df_all["title"].apply(lambda t: len(re.findall(r"\d+", str(t))))

df_all["text_number_count"] = df_all["text"].apply(lambda t: len(re.findall(r"\d+", str(t))))

df_all["text_url_count"] = df_all["text"].apply(lambda x: len(re.findall(r'http[s]?://', str(x))))

df_all["general_category"] = df_all["subject"].apply(lambda x: "Politics-news" if str(x).lower() in ["politics", "politicsnews"] else "World-news")

df_all["subject"] = df_all["subject"].replace("Government News", "Govern-News")



# Create the directory structure for the dataset file
base_path_write = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
os.makedirs(base_path_write, exist_ok=True)
write_file = os.path.join(base_path_write, "news.csv")

# Save the cleaned and processed dataframe to a CSV file
df_all.to_csv(write_file, index=False)