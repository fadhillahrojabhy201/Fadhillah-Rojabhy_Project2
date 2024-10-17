# Automated Text Summarization Using BERT: A Case Study on Indonesian News Articles with the Liputan6 Dataset

## Introduction
Text Summarization is a computational technique designed to condense lengthy documents into concise summaries while retaining the most important information. This method helps streamline the process of extracting key insights from large volumes of text, saving time and effort for individuals who need to quickly comprehend important documents. The advancement of natural language processing (NLP) techniques, such as transformer models like BERT, has significantly improved the accuracy and efficiency of automated summarization systems.

In this project, we aim to develop an automated text summarization system using the BERT model, which is pre-trained on large corpora of text to generate coherent and informative summaries. The system is designed to assist business executives, researchers, and professionals by reducing the amount of time spent reading and processing complex documents.

## Objective
The primary objective of this project is to experiment with a BERT-based model to perform text summarization on a dataset of news articles. The system will automatically generate concise summaries of the original text, maintaining the essential information while significantly reducing the text length. 

## Dataset
The dataset utilized in this project is the Liputan6 Dataset, which comprises a collection of Indonesian news articles sourced from the Liputan6 website. This dataset serves as the core resource for training and evaluating text summarization models. Each record in the dataset is structured with the following fields:
1. id: A unique identifier for each article.
2. url: The web link to the news article.
3. clean_article: The full content of the news article, which has been preprocessed to remove unnecessary elements such as advertisements or formatting tags.
4. clean_summary: An abstractive summary of the article, generated either manually or automatically.
5. extractive_summary: Index-based extractive summaries, which highlight key sentences from the original text.

The project relies on the id_liputan6 dataset, which is available on the Hugging Face platform. This dataset, specifically curated from Liputan6 — a major Indonesian news source — provides rich text-summarization pairs. These pairs are crucial for training models that can perform text summarization in the Indonesian language.

The id_liputan6 dataset is divided into two main subsets:
1. Xtreme Set: This set is designed for challenging summarization tasks, containing a diverse range of article lengths and complexities. It pushes the limits of summarization models by requiring them to summarize intricate and longer articles. This set is used to test the robustness and capability of models in more complex scenarios.
2. Canonical Set:
    a. The canonical set serves as the primary dataset for training and benchmarking. It offers a larger collection of news articles and their summaries, providing a balanced representation of various categories such as politics, entertainment, and sports. This set is ideal for training general summarization models, as it reflects the diversity of mainstream Indonesian news reporting.
    b. The Canonical Set covers a comprehensive range of topics and writing styles typical of Indonesian news, ensuring that the model is trained on a rich and varied corpus. This helps the model learn the nuances of the language and adapt to different contexts.

By using the Canonical Subset, the Bert2GPT model is exposed to a wide variety of news content during the fine-tuning process. This ensures the model can accurately generate concise, coherent, and informative summaries that remain faithful to the original text. The choice of this subset is strategic, allowing the model to be highly relevant and applicable across different types of Indonesian news articles.

## Methodology
### Exploratory Data Analysis (EDA)
The Exploratory Data Analysis (EDA) phase is critical in understanding the dataset's underlying patterns, structure, and characteristics. In this project, EDA offers insights into the id_liputan6 dataset, guiding the development and fine-tuning of the Bert2GPT Indonesian Text Summarizer. The following analytical steps were taken to provide a comprehensive understanding of the dataset:
#### 1. Distribution of Article Lengths
This step examines the lengths of articles to understand the overall distribution within the dataset. By plotting a histogram, we can observe variations in article length and detect potential outliers or biases in the data. This analysis helps determine if certain article lengths dominate the dataset, which may impact the model’s performance.
![image1](https://github.com/user-attachments/assets/ca8dd9e7-7171-4769-aa2f-b750c1aadf48)
The Distribution of Article Lengths in the dataset reveals that the majority of articles are relatively short, with 34.43% of them having fewer than 1,000 characters. The mean article length is approximately 1,407 characters, with a median of 1,185 characters, indicating that most articles fall between 896 and 1,661 characters. There are a few outliers, with 0.04% of articles exceeding 10,000 characters, creating a strong positive skew (4.42) and high kurtosis (72.91). This skewness suggests that while most articles are concise, there are rare but significant outliers that are much longer.
These findings highlight the importance of training the summarization model to handle a wide range of article lengths, from very short to exceptionally long.

#### 2. Distribution of Summary Lengths
Similarly, I analyze the distribution of summary lengths to understand the patterns in the summarization process. This step is essential to ensure that the generated summaries are concise while still retaining key information. A histogram is used to visualize how summary lengths vary, aiding in fine-tuning the model to produce balanced and informative summaries.
![image2](https://github.com/user-attachments/assets/1d920f98-821c-4753-8b07-8dab6ef77399)
The histogram shows a relatively normal distribution, with a peak around 170-210 characters, indicating that most summaries are concise and fall within a tight range. The curve tails off sharply after 300 characters, indicating that very long summaries are rare.
The dataset is consistent in terms of summary lengths, with most summaries being short and informative. The distribution's tight range suggests that the model will focus on generating concise summaries around 190 characters, with minimal outliers or extreme summary lengths. This information is important for training the model to generate summaries that are concise, aligned with the dataset, and relevant.


#### 3. Most Frequent Terms in Articles
This step identifies the most frequent terms in the original articles. By visualizing these terms through a bar plot, we gain insight into the common themes and topics that dominate the dataset. This information helps in understanding the general context of the articles and aligning the model to prioritize relevant terms during summarization.
![image3](https://github.com/user-attachments/assets/e7de7f3a-29e3-4fbf-84d7-79e697b96bfb)
<b>Top 5 Terms:</b>
1. yang: 887,893 occurrences
2. di: 859,169 occurrences
3. dan: 668,724 occurrences
4. dengan: 297,543 occurrences
5. dari: 297,440 occurrences

<b>Key Observations:</b>
Common Words: The most frequent terms are largely common Indonesian conjunctions, prepositions, and pronouns, such as yang, di, dan, and dengan. These are function words that are essential for sentence structure but do not carry specific topical meaning.
Liputan6.com: The term "Liputan6.com" appears frequently (179,903 occurrences), which is expected given that the dataset is sourced from this platform.
Negations and Context Words: Words like tidak (no/not) and tak (not) also appear frequently, indicating that many articles involve statements of negation or contrasting ideas.

<b>Thematic Terms:</b>
Some thematic terms reflect common subjects in news articles, such as warga (citizens), menjadi (become), and para (indicating plural people, like citizens or officials).
This frequency distribution highlights the need for the summarization model to focus less on these common function words and more on terms that carry significant meaning, such as names, events, or specific actions, when generating summaries. This can help ensure that the summaries are more informative and less dominated by non-essential words.

#### 4. Most Frequent Terms in Summaries
This analysis mirrors the frequent term analysis for articles but focuses on the summaries. The frequent terms in the summaries provide insights into which parts of the articles are being prioritized in the summarization process. A bar plot is used to showcase the top terms, which help in refining the model to generate meaningful and focused summaries.
![image4](https://github.com/user-attachments/assets/6556c2b6-abab-41e2-a914-d7434ced4307)
The analysis of the Most Frequent Terms in Summaries provides a view of the most common words used in the summaries of the articles. Below are the key insights based on the top 30 terms:

<b>Top 5 Terms:</b>
1. di: 151,292 occurrences
2. yang: 105,199 occurrences
3. dan: 88,776 occurrences
4. dari: 38,151 occurrences
5. dengan: 36,581 occurrences

<b>Key Observations:</b>
Common Function Words: Similar to the articles, the most frequent terms are common function words like di (in), yang (which/who), dan (and), and dari (from), which are essential for structuring sentences.
Thematic Terms: Some meaningful content terms, such as warga (citizens), rumah (house), and korban (victim), provide insight into the nature of the news topics, indicating frequent discussions about societal issues, disasters, or incidents.
Numbers and Currency: The presence of terms like dua (two) and Rp (Rupiah, Indonesian currency) indicates the frequent mention of quantities and financial information in the summaries.

<b>Comparison to Articles:</b>
In both articles and summaries, common words (di, yang, dan) dominate the list, reflecting their general role in language construction.
The summaries tend to condense the essence of the articles while retaining important thematic words, such as warga and korban, ensuring that key subjects are highlighted.

<b>Conclusion:</b>
The frequent use of function words suggests that summaries retain key sentence structures similar to the articles. However, the inclusion of more meaningful content-specific terms like warga, rumah, and korban ensures that summaries focus on the core topics discussed in the articles. This is important for ensuring that the model generates summaries that are not only concise but also retain the most relevant content.

#### 5. Scatter Plot for Article vs. Summary Length
A scatter plot is used to compare the lengths of the original articles and their corresponding summaries. This visualization provides an overview of the relationship between the article length and the compression ratio during summarization. It helps in analyzing how well the model balances between retaining information and reducing text length.
![image5](https://github.com/user-attachments/assets/df908f87-a7c8-46eb-b529-b6956ad81bbd)

The Scatter Plot for Article vs. Summary Length provides insights into the relationship between the length of the articles and their corresponding summaries. Here are the key observations:

<b>Key Statistics:</b>
Correlation Coefficient: 0.13, indicating a weak positive correlation between the article length and summary length. This suggests that while there is some relationship between longer articles and longer summaries, it is not particularly strong.
Average Compression Ratio: 0.17, meaning on average, summaries are about 17% the length of the original articles. This indicates that the summarization process significantly reduces the amount of text while retaining key information.

<b>Visual Interpretation:</b>
Main Cluster: The majority of articles, particularly those between 0 and 10,000 characters in length, are summarized in a relatively tight range between 100 and 300 characters. This shows that most summaries are consistent in length, regardless of article size.
Outliers: A few outliers represent extremely long articles (up to 40,000 characters) and corresponding summaries that are also longer but still within the typical range (100–500 characters). These outliers are rare but demonstrate the ability of the dataset to handle longer texts.

<b>Conclusion:</b>
The weak correlation and relatively consistent summary length suggest that the summarization process is effective in compressing information, regardless of the article's length. This ensures that even longer articles can be summarized concisely, while shorter articles are not overly truncated. The average compression ratio (0.17) further emphasizes that the summaries are succinct, reducing the article length significantly while retaining the core information.


#### 6. Box Plot for Distribution of Text Lengths
The box plot offers a visual representation of the variance and central tendencies in text lengths for both articles and summaries. This helps to understand the range of lengths the model will need to handle and ensures that outliers or excessively long articles do not skew the model's training.
![image6](https://github.com/user-attachments/assets/04524d59-ed1b-4f83-9f7a-fe297b88afe7)
The Box Plot for Distribution of Text Lengths compares the lengths of articles and summaries, providing insights into the range and distribution of both text types. Here’s a summary of the key results:
a. Article Lengths: The distribution of article lengths shows a larger range, with a median of around 1,185 characters. There are several extreme outliers (articles longer than 10,000 characters), as indicated by the dots beyond the upper whisker.
b. Summary Lengths: The summary lengths have a tighter distribution, with most summaries falling between 170 and 210 characters. There are fewer outliers, and the maximum length is 586 characters, which is much shorter compared to the longest articles.

<b>Conclusion:</b>
The box plot clearly shows that summaries are significantly shorter than the original articles, with a more consistent length distribution. Articles have a wider range of lengths, with a few very long outliers, while summaries are typically concise and within a narrow range. This visual reinforces the fact that the summarization process reduces the text length significantly while keeping the content concise.


#### 7. Plotting Bi-grams in Articles
Bi-grams, which are pairs of consecutive words, are plotted to reveal common phrase patterns in the articles. Analyzing bi-grams helps understand the contextual relationships between words and provides insights into frequently occurring phrases that the model can prioritize during summarization.
![image7](https://github.com/user-attachments/assets/df11b389-c019-4c13-b34a-0321be430aa6)
The Top Bi-grams in Articles analysis shows the most frequently occurring pairs of words (bi-grams) in the articles.The Top Bi-grams in Articles analysis shows the most frequently occurring pairs of words (bi-grams) in the articles. Here’s a summary of the key results:
1. Frequent References to Liputan6: The bi-grams "liputan6 com," "liputan sctv," and "tim liputan" occur frequently because the dataset is sourced from the Liputan6 news platform. These bi-grams reflect references to the news outlet, reporters, and broadcasting teams.
2. Location and Time References: Phrases like di jakarta (in Jakarta) and saat ini (at this time) are common, indicating frequent mentions of the location Jakarta and time-sensitive news reporting.
3. Healthcare and Disaster Terms: Rumah sakit (hospital) appears frequently, indicating common coverage of health-related topics or events involving hospitals.
4. Geographic Regions: Bi-grams such as jawa barat (West Java), jawa timur (East Java), and jawa tengah (Central Java) show that the news often covers events in various regions of Indonesia.
5. General Terms: Bi-grams like baru ini (just now), salah satu (one of), and sementara itu (meanwhile) are often used to structure news stories, indicating time and comparisons.

<b>Conclusion:</b>
The most frequent bi-grams in the dataset reflect the recurring mention of the news outlet Liputan6, along with common geographical and time-related references. The frequent usage of healthcare terms and regional locations suggests coverage of health and regional events. These insights are useful for training the summarization model to recognize and prioritize key contextual information in the articles.


#### 8. Plotting Bi-grams in Summaries
Bi-gram analysis is extended to summaries to uncover the most common two-word sequences. This step helps in refining the model’s ability to capture and reproduce key phrases in the summaries that are crucial for maintaining coherence and context.
![image8](https://github.com/user-attachments/assets/21fd7f60-8109-44be-80ce-e3b974e987ba)
The Top Bi-grams in Summaries show the most frequently occurring pairs of words in the summaries of the news articles. Here’s a summary of the key results:
1. Geographical Terms: The bi-grams di kawasan (in the area), di jakarta (in Jakarta), jawa barat (West Java), jawa timur (East Java), and jawa tengah (Central Java) suggest that summaries often emphasize locations and regional references, which are important for identifying where events took place.
2. Healthcare Terms: Rumah sakit (hospital) is a frequently mentioned bi-gram, indicating the prevalence of healthcare-related stories, likely involving hospitalizations or medical facilities.
3. Time References: Phrases like hari ini (today) and saat ini (at this time) show that many summaries focus on timely, current events.
4. Protests and Incidents: Berunjuk rasa (protest) and terjadi di (occurred in) suggest that the summaries often cover incidents, including protests or events with significant public attention.
5. Contextual References: Phrases like salah satu (one of) and yang akan (which will) are often used to give context or indicate future actions.

<b>Conclusion:</b>
The frequent bi-grams in the summaries provide a condensed focus on important aspects such as location, healthcare, incidents, and time-sensitive events. These bi-grams reflect the core details summarized from the articles, highlighting key elements such as where, when, and what occurred, which is critical for concise news reporting. This pattern helps the model understand which content is essential for summarization and improves its ability to generate relevant and meaningful summaries.


#### 9. Plotting Tri-grams in Articles
The tri-gram analysis focuses on frequent three-word sequences in the articles. By identifying these tri-grams, we gain deeper insights into contextual word relationships, which can help the model generate more accurate and coherent summaries.
![image9](https://github.com/user-attachments/assets/1988a9bf-3f4e-4aa3-bbe3-b6af6de28f8a)
The Top Tri-grams in Articles analysis highlights the most frequent three-word combinations (tri-grams) found in the articles. Here’s a summary of the key results:
1.References to the News Outlet: The most frequent tri-grams, liputan6 com jakarta and tim liputan sctv, are related to the Liputan6 news platform and its reporting teams, which is expected given the source of the dataset.
2. Recent Events: Baru baru ini (recently) is commonly used to discuss current events, indicating the time-sensitive nature of the articles.
3. Healthcare: Phrases like di rumah sakit (in the hospital) and ke rumah sakit (to the hospital) suggest frequent reporting on healthcare or incidents involving hospitals.
4. Political and Government Figures: Susilo bambang yudhoyono (a former president of Indonesia) and presiden susilo bambang indicate that there is significant coverage of political figures.
5. Geographic and Political Terms: Tri-grams like nanggroe aceh darussalam (a province in Indonesia), komisi pemilihan umum (General Election Commission), and bahan bakar minyak (fuel oil) suggest the coverage of political, regional, and economic topics.

<b>Conclusion:</b>
The tri-grams reflect a mix of content related to news reporting from Liputan6, healthcare, political figures, and regions in Indonesia. These frequent phrases show the importance of events and figures in the Indonesian political and healthcare landscapes, which are often covered in the news. Understanding these frequent tri-grams helps in building better summarization models, as they indicate recurring themes and content that should be highlighted in summaries.


#### 10. Plotting Tri-grams in Summaries
As with the articles, tri-gram analysis is applied to the summaries to understand which three-word phrases are most commonly retained in the summarization process. This informs the model on how to reproduce important multi-word phrases that carry significant meaning.
![image10](https://github.com/user-attachments/assets/6d00cbcc-2c48-4140-9e71-17bea32ce0a5)
The Top Tri-grams in Summaries analysis identifies the most frequent three-word combinations in the summaries of news articles. Here’s a summary of the key results:
1. Political Figures: The most frequent tri-gram, susilo bambang yudhoyono, refers to the former Indonesian president, showing the significant focus on political figures in news summaries. Presiden susilo bambang and presiden megawati sukarnoputri (another former president) further emphasize the focus on politics.
2. Healthcare Terms: Di rumah sakit (in the hospital) is one of the most common tri-grams, indicating frequent coverage of healthcare topics or incidents involving hospitals.
3. Protests and Incidents: Berunjuk rasa di (protest at/in) and ada korban jiwa (there were fatalities) suggest that the news often covers protests and incidents with significant consequences.
4. Economic and Fuel Issues: Bahan bakar minyak (fuel oil) and kenaikan harga bbm (increase in fuel prices) point to coverage of economic issues, particularly fuel prices.
5. Sports Figures: Sir alex ferguson appears in the top tri-grams, indicating coverage of international sports, particularly related to football.

<b>Conclusion:</b>
The frequent tri-grams in summaries reflect a strong focus on political figures, healthcare incidents, protests, and economic issues. These tri-grams show the core topics that are often highlighted in news summaries, which helps to direct the summarization model's attention towards important events, figures, and topics.


#### 11. Sentiment Analysis for Articles
Sentiment analysis is conducted on the original articles to determine the overall emotional tone and sentiment polarity (positive, negative, or neutral). Understanding the sentiment of the articles helps ensure that the summarization process maintains the original emotional context and tone.

![image11](https://github.com/user-attachments/assets/f3fde216-e379-41e2-a267-06b7be064378)
The histogram shows a significant spike at a sentiment polarity of 0, which corresponds to the high proportion of neutral articles. A smaller distribution is observed for both positive and negative sentiments, with very few articles showing strong sentiment in either direction (extreme values like -1.0 or 1.0).
The Sentiment Analysis for Articles provides an overview of the sentiment polarity in the news articles. Here are the key results:
1. Statistical Overview:
    1a. Total Articles Analyzed: 204,855
    1b. Mean Sentiment Score: 0.022, indicating that the overall sentiment leans slightly positive.
    1c. Standard Deviation: 0.120, showing low variability in sentiment scores across articles.
    1d. Minimum Score: -1.0 (indicating extremely negative sentiment)
    1e. Maximum Score: 1.0 (indicating extremely positive sentiment)
    1f. Median Sentiment Score: 0.0 (the majority of articles are neutral)
2. Sentiment Distribution:
    2a. Positive Sentiment: 14.60% of the articles have positive sentiment scores.
    2b. Negative Sentiment: 4.81% of the articles are classified as having negative sentiment.
    2c. Neutral Sentiment: The majority of the articles (80.59%) are neutral, with a sentiment score close to 0.

<b>Conclusion:</b>
The dataset is overwhelmingly neutral, with 80.59% of the articles falling into this category. This suggests that the news articles typically provide factual reporting rather than subjective or opinionated content. A relatively small proportion of articles carry a positive or negative tone, which is important to note when training models for sentiment analysis. The slight positive skew in the dataset's mean (0.022) indicates that there are more positive articles than negative ones overall.


#### 12. Sentiment Analysis for Summaries
Extending sentiment analysis to the summaries ensures that the generated summaries preserve the sentiment of the original articles. This step is important for maintaining the integrity and emotional tone of the content.
![image12](https://github.com/user-attachments/assets/4a7b1942-dd1f-49a4-a1b6-c99e35181219)
The histogram shows a significant peak at a sentiment polarity of 0, reflecting the overwhelming majority of neutral summaries. A very small portion of summaries shows positive or negative sentiment, with only slight deviations from neutrality.
The Sentiment Analysis for Summaries gives an overview of the sentiment polarity in the summaries of the news articles. Here are the key results:
1. Statistical Overview:
    1a. Total Summaries Analyzed: 204,855
    1b. Mean Sentiment Score: 0.006953, indicating that overall sentiment in the summaries is very close to neutral.
    1c. Standard Deviation: 0.063, showing low variability in sentiment scores.
    1d. Minimum Score: -1.0 (extremely negative sentiment)
    1e. Maximum Score: 1.0 (extremely positive sentiment)
    1f. Median Sentiment Score: 0.0 (most summaries are neutral)
2. Sentiment Distribution:
    2a. Positive Sentiment: 3.55% of the summaries have a positive sentiment score.
    2b. Negative Sentiment: 0.93% of the summaries exhibit negative sentiment.
    2c. Neutral Sentiment: The vast majority of the summaries (95.51%) are neutral in tone.

<b>Conclusion:</b>
The sentiment analysis of summaries reveals that nearly all the summaries are neutral, with 95.51% having a sentiment score around 0. This indicates that the summarization process retains a factual and neutral tone, similar to the articles themselves. With such a small portion of summaries carrying a positive or negative sentiment, the dataset is primarily focused on delivering objective news information rather than opinionated or emotionally charged content. This neutrality is important for maintaining the integrity of news reporting in the summarization process.


#### 13. Top Words in Topics
Topic modeling, such as Latent Dirichlet Allocation (LDA), is used to uncover the main themes and topics within the dataset. By identifying the top words in various topics, we can understand the content diversity and ensure the model is well-tuned to represent the range of subjects present in the news articles.

Topic 1: yang, dan, ini, dari, itu, dengan, untuk, aceh, dalam, ke
Topic 2: yang, dan, dengan, dari, untuk, itu, ini, tidak, ke, akan
Topic 3: yang, dan, ini, warga, dari, untuk, itu, com, liputan6, dengan
Topic 4: yang, dan, itu, ini, jakarta, dalam, akan, untuk, dengan, dari
Topic 5: yang, dan, ini, itu, polisi, dengan, dari, tak, jakarta, mereka
Each step in the EDA process provides a clearer understanding of the dataset's structure, informing the development and fine-tuning of the Bert2GPT Indonesian Text Summarizer. These insights ensure the model can generate concise, coherent, and contextually accurate summaries across a broad range of news topics.
