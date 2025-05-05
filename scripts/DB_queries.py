import pandas as pd
import sqlite3
import os

# Create the directory structure for the database file
base_path_query = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
os.makedirs(base_path_query, exist_ok=True)
db_file = os.path.join(base_path_query, "news_dataset.db")

# Connect to the SQLite database
conn = sqlite3.connect(db_file)

# Create the directory structure for the queries file
base_path_query = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "queries"))
os.makedirs(base_path_query, exist_ok=True)



query0 = """
    SELECT * 
    FROM news 
    LIMIT 5;
"""
# Execute the query and read the results into a dataframe
df0 = pd.read_sql_query(query0, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query0.csv")
df0.to_csv(data_file, index=False)



# Are fake news titles more emotionally charged than true ones?
query1 = """
    SELECT label, AVG(title_emotional_word_count) AS avg_emotion
    FROM news
    GROUP BY label;
"""

# Execute the query and read the results into a dataframe
df1 = pd.read_sql_query(query1, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query1.csv")
df1.to_csv(data_file, index=False)



# Are fake news titles more often written as questions?
query2 = """
    SELECT label, AVG(is_question_title) AS question_ratio
    FROM news
    GROUP BY label;
"""

# Execute the query and read the results into a dataframe
df2 = pd.read_sql_query(query2, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query2.csv")
df2.to_csv(data_file, index=False)



# Do fake news articles contain more hyperlinks than true news?
query3 = """
    SELECT label, AVG(text_url_count) AS avg_links
    FROM news
    GROUP BY label;
"""

# Execute the query and read the results into a dataframe
df3 = pd.read_sql_query(query3, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query3.csv")
df3.to_csv(data_file, index=False)



# Do fake news articles show lower lexical diversity (simpler language)?
query4 = """
    SELECT label, AVG(text_lexical_diversity) AS avg_diversity
    FROM news
    GROUP BY label;
"""

# Execute the query and read the results into a dataframe
df4 = pd.read_sql_query(query4, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query4.csv")
df4.to_csv(data_file, index=False)



# Are longer titles more likely to be fake?
query5 = """
    SELECT label, COUNT(*) AS count
    FROM news
    WHERE title_word_count > 12
    GROUP BY label;
"""

# Execute the query and read the results into a dataframe
df5 = pd.read_sql_query(query5, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query5.csv")
df5.to_csv(data_file, index=False)



# What percentage of fake vs. true news titles include numbers?
query6 = """
  SELECT label, ROUND(SUM(CASE WHEN title_number_count > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS percent_with_number
  FROM news
  GROUP BY label;
"""

# Execute the query and read the results into a dataframe
df6 = pd.read_sql_query(query6, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query6.csv")
df6.to_csv(data_file, index=False)



# Do fake news titles use a higher ratio of stopwords in their text?
query7 = """
    SELECT label, AVG(text_stopword_ratio) AS avg_stop_ratio
    FROM news
    GROUP BY label;
"""

# Execute the query and read the results into a dataframe
df7 = pd.read_sql_query(query7, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query7.csv")
df7.to_csv(data_file, index=False)



# Which subject has the highest proportion of fake news?
query8 = """
    SELECT general_category,
        COUNT(*) AS total_articles,
        SUM(CASE WHEN label = 'fake' THEN 1 ELSE 0 END) AS fake_count,
        ROUND(SUM(CASE WHEN label = 'fake' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS fake_percentage
    FROM news
    GROUP BY general_category
    ORDER BY fake_percentage DESC
    LIMIT 5;
"""

# Execute the query and read the results into a dataframe
df8 = pd.read_sql_query(query8, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query8.csv")
df8.to_csv(data_file, index=False)



# On which days of the week is the publication of fake news more likely?
query9 = """
    SELECT
        strftime('%w', date) AS weekday_number,
        CASE strftime('%w', date)
            WHEN '6' THEN 'Sunday'
            WHEN '0' THEN 'Monday'
            WHEN '1' THEN 'Tuesday'
            WHEN '2' THEN 'Wednesday'
            WHEN '3' THEN 'Thursday'
            WHEN '4' THEN 'Friday'
            WHEN '5' THEN 'Saturday'
        END AS weekday_name,
        label,
        COUNT(*) AS article_count
    FROM news
    WHERE date IS NOT NULL
    GROUP BY weekday_number, label
    ORDER BY weekday_number;
"""

# Execute the query and read the results into a dataframe
df9 = pd.read_sql_query(query9, conn)

# Save the query result in a dataframe
data_file = os.path.join(base_path_query, "query9.csv")
df9.to_csv(data_file, index=False)