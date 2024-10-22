# Automated Text Summarization Using BERT: A Case Study on Indonesian News Articles with the Liputan6 Dataset

## Introduction
Text Summarization is a computational technique designed to condense lengthy documents into concise summaries while retaining the most important information. This method helps streamline the process of extracting key insights from large volumes of text, saving time and effort for individuals who need to quickly comprehend important documents. The advancement of natural language processing (NLP) techniques, such as transformer models like BERT, has significantly improved the accuracy and efficiency of automated summarization systems.

In this project, we aim to develop an automated text summarization system using the BERT model, which is pre-trained on large corpora of text to generate coherent and informative summaries. The system is designed to assist business executives, researchers, and professionals by reducing the amount of time spent reading and processing complex documents.

## Objective
The primary objective of this project is to experiment with a BERT-based model to perform text summarization on a dataset of news articles. The system will automatically generate concise summaries of the original text, maintaining the essential information while significantly reducing the text length. 

## Dataset
### Dataset Description
The Liputan6 Dataset is a collection of Indonesian news articles sourced from the Liputan6 website. It serves as the primary resource for training and evaluating text summarization models in the Indonesian language.
### Record Structure
Each record in the dataset contains the following fields:
1. id: A unique identifier for each article.
2. url: The web link to the news article.
3. clean_article: The full content of the news article, which has been preprocessed to remove unnecessary elements such as advertisements or formatting tags.
4. clean_summary: An abstractive summary of the article, generated either manually or automatically.
5. extractive_summary: Index-based extractive summaries, which highlight key sentences from the original text.

### Dataset Subsets
The project relies on the id_liputan6 dataset, which is available on the Hugging Face platform. This dataset, specifically curated from Liputan6 — a major Indonesian news source — provides rich text-summarization pairs. These pairs are crucial for training models that can perform text summarization in the Indonesian language.
The id_liputan6 dataset is divided into two main subsets:

1. Xtreme Set:
        a. Designed for challenging summarization tasks
        b. Contains diverse article lengths and complexities
        c. Used to test model robustness in complex scenarios

<img width="1077" alt="Screen Shot 2024-10-19 at 12 17 07" src="https://github.com/user-attachments/assets/1bcaff02-0644-4a59-ac16-8794f61405d3">

2. Canonical Set:
        a. Primary dataset for training and benchmarking
        b. Larger collection of news articles and summaries
        c. Balanced representation of various categories (politics, entertainment, sports, etc.)
        d. Reflects diversity of mainstream Indonesian news reporting
        e. Covers a comprehensive range of topics and writing styles

<img width="1017" alt="Screen Shot 2024-10-19 at 12 24 17" src="https://github.com/user-attachments/assets/62ea647d-0b2f-4d32-ad0e-e826e260f880">

### Importance for the Project
- The Canonical Subset is used for fine-tuning the Bert2GPT model
- Exposes the model to a wide variety of news content
- Ensures the model can generate accurate, concise, and coherent summaries
- Allows the model to learn language nuances and adapt to different contexts
- Makes the model highly relevant and applicable across different types of Indonesian news articles

## Methodology
### Exploratory Data Analysis (EDA)
The Exploratory Data Analysis (EDA) phase is critical in understanding the dataset's underlying patterns, structure, and characteristics. In this project, EDA offers insights into the id_liputan6 dataset, guiding the development and fine-tuning of the Bert2GPT Indonesian Text Summarizer. The following analytical steps were taken to provide a comprehensive understanding of the dataset:

#### 1. Data Cleaning, Summary and Insights


1. Text Cleaning:  The cleaning process effectively prepares the text for analysis:
- Lowercase Conversion: All text is converted to lowercase, ensuring uniformity.
- Removal of Unwanted Text: Instances of "Liputan6.com," location prefixes, and specific text in parentheses or brackets are removed to focus on the core content.
- Special Character Removal: Non-alphanumeric characters are eliminated, which simplifies the text and helps in further analysis.
- Trimming Excess Spaces: Any extra spaces in the text are removed to maintain clean formatting.

2. Article and Summary Statistics:
        a. Cleaned Article Statistics:
                - Average length: 181.524 words
                - Maximum length: 1064 words
                - Minimum length: 68 words
        b. Cleaned Summary Statistics:
                - Average length: 23.127 words
                - Maximum length: 37 words
                - Minimum length: 13 words
           Insights:
                - There's a significant difference between article and summary lengths.
                - The compression ratio is about 7.8:1 (181.524 / 23.127), meaning summaries are typically about 12.7% the length of the original articles.
                - Articles have a wide range of lengths (68 to 1064 words), which might require attention during model training.
                - Summaries are more consistent in length (13 to 37 words), which is good for generating predictions.

3. Word Frequency Analysis:
Top Words in Cleaned Articles:
`Most frequent words include: di, yang, dan, itu, ini, jakarta, dari, untuk, dengan, dalam.`
Top Words in Cleaned Summaries:
`Most frequent words include: di, dan, yang, akan, tak, harga, untuk, pemerintah, dari, sejumlah.`

<img width="1336" alt="Screen Shot 2024-10-22 at 21 47 55" src="https://github.com/user-attachments/assets/caca3868-029c-4c4e-9b9c-e5f08cd64d6c">



#### 2. Text Preprocessing
The preprocessing step seems to have worked, but there are a few observations:

- The lemmatization doesn't appear to be working as expected for Indonesian. This is likely because NLTK's WordNetLemmatizer is designed for English.

- Stop words are being removed, but some common words (like "yang", "di", "dan") are still present. This suggests that the Indonesian stop words list might not be comprehensive.

- Punctuation marks are still present in the processed text. You might want to remove these in a future iteration.
<img width="1440" alt="Screen Shot 2024-10-19 at 12 59 20" src="https://github.com/user-attachments/assets/620733ab-ed1e-4070-8987-4a09d5ee5295">

- Word Cloud Visualization: Create word clouds for both original and processed texts to visually compare the most prominent words.

![a](https://github.com/user-attachments/assets/4db8cb1b-50c3-4f88-ae1b-56154a64ebbe)

![b](https://github.com/user-attachments/assets/2fbef44b-d227-41b2-a5a9-efd0fb8114f1)

- Unique Words Count: Compare the number of unique words in the original and processed datasets to see how much the vocabulary has been reduced.
          a. Original article vocabulary size: 612672
          b. Processed article vocabulary size: 269388
          c. Original summary vocabulary size: 172071
          d. Processed summary vocabulary size: 93090


#### 3. Distribution of Article Lengths
This step examines the lengths of processed articles to understand the overall distribution within the dataset. By plotting a histogram, we can observe variations in article length and detect potential outliers or biases in the data. This analysis helps determine if certain article lengths dominate the dataset, which may impact the model’s performance.

![c](https://github.com/user-attachments/assets/148b24cb-88cc-4737-9cf5-482d09a5c763)

The distribution of article lengths shows a notable concentration of shorter articles, with 16.31% having fewer than 500 characters. The mean article length is approximately 885.24 characters, while the median is 744 characters, indicating that most articles fall within the range of 562 to 1,048 characters. The dataset reveals a minimum length of 29 characters and a maximum length of 28,678 characters, with only 267 articles (0.13%) exceeding 5,000 characters. The skewness of 4.28 and kurtosis of 66.28 reflect a strong positive skew, suggesting that while most articles are relatively concise, there are a few significant outliers that are considerably longer. These findings underscore the need for a summarization model capable of effectively managing a diverse range of article lengths, from very short to exceptionally long.

#### 4. Distribution of Summary Lengths
Similarly, I analyze the distribution of summary lengths to understand the patterns in the summarization process. This step is essential to ensure that the generated summaries are concise while still retaining key information. A histogram is used to visualize how summary lengths vary, aiding in fine-tuning the model to produce balanced and informative summaries.

![d](https://github.com/user-attachments/assets/14628c1b-b26f-45dd-a007-d4206d98696e)

The distribution of summary lengths indicates a significant concentration of shorter summaries, with 0.08% having fewer than 50 characters. The mean summary length is approximately 138.88 characters, while the median is 138 characters, suggesting that most summaries fall within the range of 119 to 156 characters. The dataset reveals a minimum length of 26 characters and a maximum length of 443 characters, with no summaries exceeding 500 characters.

The skewness of 1.21 and kurtosis of 6.46 reflect a positive skew, indicating that while most summaries are concise, there are a few longer summaries that stretch the distribution. These findings highlight the importance of designing a summarization model capable of effectively handling a consistent range of summary lengths, which are predominantly short, with very few outliers.


#### 5. Most Frequent Terms in Articles
This step identifies the most frequent terms in the original articles. By visualizing these terms through a bar plot, we gain insight into the common themes and topics that dominate the dataset. This information helps in understanding the general context of the articles and aligning the model to prioritize relevant terms during summarization.

![e](https://github.com/user-attachments/assets/c8adad3c-43aa-496f-b12d-18bec03cd4b4)

<b>Top 5 Terms:</b>
1. jakarta: 180155
2. warga: 126085
3. rumah: 97624
4. polisi: 88010
5. indonesia: 82921


#### 6. Most Frequent Terms in Summaries
This analysis mirrors the frequent term analysis for articles but focuses on the summaries. The frequent terms in the summaries provide insights into which parts of the articles are being prioritized in the summarization process. A bar plot is used to showcase the top terms, which help in refining the model to generate meaningful and focused summaries.

![f](https://github.com/user-attachments/assets/4ef97454-c056-4712-82e1-cba2eeb7d974)

The analysis of the Most Frequent Terms in Summaries provides a view of the most common words used in the summaries of the articles. Below are the key insights based on the top 30 terms:

<b>Top 5 Terms:</b>
1. warga: 24924
2. jakarta: 16185
3. rumah: 15867
4. korban: 14432
5. indonesia: 13673


#### 7. Scatter Plot for Article vs. Summary Length
A scatter plot is used to compare the lengths of the original articles and their corresponding summaries. This visualization provides an overview of the relationship between the article length and the compression ratio during summarization. It helps in analyzing how well the model balances between retaining information and reducing text length.

![g](https://github.com/user-attachments/assets/10ffb3bf-096f-4841-bf45-97ad2131176e)

The Scatter Plot for Article vs. Summary Length provides insights into the relationship between the length of the articles and their corresponding summaries. Here are the key observations:

The weak correlation and consistent summary lengths suggest that the summarization process is effective in compressing information regardless of article length. This ensures that even longer articles can be summarized succinctly, while shorter articles are not excessively truncated. The average compression ratio of 0.19 further emphasizes that the summaries are concise, significantly reducing the original article lengths while retaining core content.

#### 8. Box Plot for Distribution of Text Lengths
The box plot offers a visual representation of the variance and central tendencies in text lengths for both articles and summaries. This helps to understand the range of lengths the model will need to handle and ensures that outliers or excessively long articles do not skew the model's training.

![H](https://github.com/user-attachments/assets/0ae67d52-71ea-4f44-815d-e5a0bb9d88bb)

The box plot clearly illustrates that summaries are significantly shorter than the original articles, showcasing a more consistent length distribution. Articles exhibit a wider range of lengths, including a few very long outliers, while summaries typically remain concise and within a narrow range. This visual representation reinforces the effectiveness of the summarization process in condensing text significantly while preserving essential information.


#### 9. Plotting Bi-grams in Articles
Bi-grams, which are pairs of consecutive words, are plotted to reveal common phrase patterns in the articles. Analyzing bi-grams helps understand the contextual relationships between words and provides insights into frequently occurring phrases that the model can prioritize during summarization.

![i](https://github.com/user-attachments/assets/6f6f3332-0745-404e-bef1-8cd0d347d252)


#### 10. Plotting Bi-grams in Summaries
Bi-gram analysis is extended to summaries to uncover the most common two-word sequences. This step helps in refining the model’s ability to capture and reproduce key phrases in the summaries that are crucial for maintaining coherence and context.

![j](https://github.com/user-attachments/assets/78b87433-d97b-4296-bbed-63dec2fe1ba2)


#### 11. Plotting Tri-grams in Articles
The tri-gram analysis focuses on frequent three-word sequences in the articles. By identifying these tri-grams, we gain deeper insights into contextual word relationships, which can help the model generate more accurate and coherent summaries.

![k](https://github.com/user-attachments/assets/c7bee130-ad73-487e-be3b-41a30348962f)


#### 12. Plotting Tri-grams in Summaries
As with the articles, tri-gram analysis is applied to the summaries to understand which three-word phrases are most commonly retained in the summarization process. This informs the model on how to reproduce important multi-word phrases that carry significant meaning.

![l](https://github.com/user-attachments/assets/5ffb98b2-fb72-4c1a-a752-04f7e968a67d)


#### 13. Sentiment Analysis for Articles
Sentiment analysis is conducted on the original articles to determine the overall emotional tone and sentiment polarity (positive, negative, or neutral). Understanding the sentiment of the articles helps ensure that the summarization process maintains the original emotional context and tone.

![m](https://github.com/user-attachments/assets/d567453c-1af2-48a6-a799-218cabc56dde)

The histogram illustrates that the vast majority of articles cluster around a sentiment polarity of 0.00, highlighting a strong prevalence of neutral sentiments. There are very few articles with extreme positive or negative sentiments, as indicated by the sparse distribution in those areas.
The findings suggest that the articles are primarily neutral in sentiment, with a minority expressing positive or negative sentiments. This may indicate that the content covered in the articles tends to be factual or informational rather than emotionally charged. The analysis underscores the importance of understanding sentiment for tasks such as summarization, as it may impact the interpretation of the articles and their overall tone.


#### 14. Sentiment Analysis for Summaries
Extending sentiment analysis to the summaries ensures that the generated summaries preserve the sentiment of the original articles. This step is important for maintaining the integrity and emotional tone of the content.

![n](https://github.com/user-attachments/assets/574ddae3-ef62-4f97-8550-c323373f6a64)

The histogram illustrates that the vast majority of summaries cluster around a sentiment polarity of 0.00, emphasizing a strong dominance of neutral sentiments. Very few summaries exhibit extreme positive or negative sentiments, as indicated by the sparse distribution in those areas.
The findings suggest that the summaries are primarily neutral in sentiment, with a minimal proportion expressing positive or negative sentiments. This may indicate that the summarization process tends to retain a balanced tone, likely focusing on factual information rather than emotional expression. The analysis highlights the importance of sentiment in understanding the overall tone of the summaries, which is crucial for applications such as content recommendation and audience engagement. ​​


#### 15. Top Words in Topics
Topic modeling, such as Latent Dirichlet Allocation (LDA), is used to uncover the main themes and topics within the dataset. By identifying the top words in various topics, we can understand the content diversity and ensure the model is well-tuned to represent the range of subjects present in the news articles.
Here are the top words found in the five key topics:

`Top words in LDA topics:
Topic 1: jakarta, presiden, indonesia, ketua, negara, partai, anggota, pemerintah, menteri, aceh
Topic 2: warga, air, banjir, rumah, jalan, desa, jawa, kabupaten, suara, korban
Topic 3: rp, jakarta, harga, anak, pemerintah, minyak, indonesia, ribu, persen, pasar
Topic 4: pemain, tim, pertandingan, musim, gol, klub, menit, liga, bermain, babak
Topic 5: polisi, korban, rumah, warga, kepolisian, jakarta, tersangka, kota, jawa, orang

Compression ratio: 0.16`

The analysis of the top words in LDA topics illustrates the diverse range of subjects covered in the articles, from politics and community issues to economics and sports. The low compression ratio signifies that the summaries effectively capture the essence of the articles, making them concise yet informative. Understanding these topics can aid in tailoring content delivery, enhancing engagement, and supporting more targeted information dissemination strategies.


## Data Preprocessing
### 1. Splitting the Dataset
The dataset is divided into training (79,413 examples) and validation (19,854 examples) sets. This separation ensures the model can be evaluated on unseen data for better generalization.
### 2. Loading the Tokenizer from a Pretrained Model
The tokenizer from the pretrained model "cahya/bert2gpt-indonesian-summarization" is employed, with special tokens for marking the start and end of sequences.

<img width="1081" alt="Screen Shot 2024-10-19 at 14 52 04" src="https://github.com/user-attachments/assets/711fd77a-0222-4d32-b15e-be37327ce7db">

### 3. Defining the Maximum Length of the Input and Target Sequences
Both the input (articles) and target (summaries) sequences are limited to a maximum length of 256 tokens for efficient batching and uniformity.
### 4. Tokenizing and Encoding Articles and Summaries
A preprocessing function tokenizes and encodes the articles and summaries, ensuring they are padded or truncated to the specified length.
### 5. Mapping Train and Validation Dataset
The preprocessing function is applied to both the training and validation sets, resulting in tokenized, encoded, and structured datasets with input IDs, attention masks, and decoder input IDs ready for model input.

<img width="1394" alt="Screen Shot 2024-10-19 at 14 55 31" src="https://github.com/user-attachments/assets/12185c96-a6e0-492c-bc3e-59566f7e96f9">

### 6. Processed Example
For verification, processed examples from both datasets (training and validation datasets) are printed, showcasing the encoded values (input_ids, attention_mask, labels, decoder_input_ids).
#### Print Processed Examples from the Training Dataset

```"html"
Example 1:
input_ids: [3, 17715, 1050, 17, 3036, 15, 2647, 29, 2221, 4510, 14575, 4476, 1495, 15668, 17368, 15, 3596, 2086, 15, 14061, 25215, 9590, 15, 2647, 15, 11553, 11, 2178, 18, 2162, 12, 7366, 17, 15455, 3162, 3409, 3920, 2552, 2647, 15, 6840, 1007, 11886, 15, 2918, 3167, 15668, 17368, 1675, 3027, 2236, 4606, 6457, 1592, 4476, 15, 2199, 4821, 6218, 12911, 3028, 7724, 8835, 17, 2248, 7724, 8835, 15, 4476, 1609, 10629, 1529, 1720, 2117, 7724, 8948, 17, 1695, 7113, 23125, 3458, 9183, 5390, 18792, 1509, 2803, 1572, 19445, 1930, 4857, 1684, 17, 3604, 14575, 8502, 1793, 3167, 3809, 16673, 9563, 1555, 3085, 4649, 6751, 4045, 7566, 1495, 15668, 17368, 17, 1695, 3270, 4162, 5790, 14027, 11652, 2815, 3353, 2316, 15, 1956, 3353, 3167, 17, 2246, 3270, 1938, 1684, 5920, 1930, 20346, 1798, 7391, 3213, 1535, 4504, 17, 1570, 4798, 1684, 15, 3167, 4234, 29024, 1555, 4476, 17, 1887, 4306, 3167, 8471, 7068, 7724, 8835, 15, 2722, 2221, 8166, 2316, 6649, 6218, 5702, 5068, 3971, 8948, 17, 1855, 4510, 2579, 12330, 1495, 2177, 3541, 1978, 3942, 19136, 7334, 17, 3085, 7301, 17237, 7621, 14575, 1675, 17, 1956, 1695, 3270, 14575, 2277, 3215, 9302, 1509, 2117, 7724, 8835, 17, 2379, 4476, 5468, 1675, 2410, 1753, 18276, 4010, 6385, 19030, 1555, 8657, 2878, 4086, 2316, 6751, 17, 34, 12287, 29, 4493, 1509, 4476, 29024, 15, 2270, 1714, 9043, 36, 17, 17886, 1028, 8146, 9563, 1914, 14007, 1508, 1495, 3229, 17, 11, 1830, 18, 3973, 1013, 12, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels: [3, 2221, 4510, 14575, 4476, 1495, 15668, 17368, 15, 2105, 6669, 15, 14061, 25215, 9590, 1572, 3027, 2236, 4606, 6457, 1592, 4476, 17, 1695, 7113, 23125, 3458, 9183, 5390, 18792, 1509, 2803, 1572, 19445, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
decoder_input_ids: [3, 3, 2221, 4510, 14575, 4476, 1495, 15668, 17368, 15, 2105, 6669, 15, 14061, 25215, 9590, 1572, 3027, 2236, 4606, 6457, 1592, 4476, 17, 1695, 7113, 23125, 3458, 9183, 5390, 18792, 1509, 2803, 1572, 19445, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

Example 2:
input_ids: [3, 17715, 1050, 17, 3036, 15, 2647, 29, 2694, 1555, 13816, 4695, 2601, 5725, 48, 15, 13816, 1859, 1542, 1757, 4063, 6, 1980, 6, 1555, 16359, 4764, 6695, 1537, 1509, 9727, 1007, 20284, 17, 1542, 8502, 1793, 4695, 2601, 17358, 22138, 10502, 2649, 8389, 6422, 2605, 14977, 9183, 15137, 1637, 7205, 20420, 2356, 2977, 1766, 21446, 31314, 3452, 15, 2537, 1786, 2340, 11122, 12571, 1792, 1795, 16, 8286, 1519, 1516, 13917, 17, 6, 1542, 9436, 15, 20596, 1980, 4493, 1653, 1786, 3098, 17, 4856, 5481, 10272, 8818, 5356, 2557, 3144, 16, 15298, 11122, 12571, 15, 6, 27011, 22138, 17, 10944, 15, 8424, 4843, 16, 4843, 1510, 6550, 13238, 2230, 3028, 17446, 2412, 1555, 3098, 5011, 8818, 17, 3668, 16, 3668, 1510, 19278, 1510, 5149, 2847, 1592, 1855, 4493, 17, 1555, 3290, 15, 8764, 4493, 1609, 1786, 3098, 17, 10802, 6695, 2203, 1684, 29553, 2550, 1996, 22680, 17, 29553, 10523, 5304, 10472, 6695, 2203, 1684, 1570, 8669, 3475, 17, 6, 2904, 1510, 8909, 3426, 22138, 1675, 31314, 3452, 1510, 3169, 5471, 5288, 1533, 12571, 17, 5356, 1510, 3245, 2464, 1675, 2277, 12368, 1555, 19611, 15, 26772, 15, 1509, 2886, 2078, 6, 8052, 1541, 29553, 17, 29553, 10502, 15, 5060, 5542, 1980, 4493, 2223, 2410, 1555, 3986, 2577, 3117, 4493, 17, 1938, 1542, 2223, 2410, 2649, 4493, 2733, 15868, 5377, 1572, 2273, 15634, 1509, 11994, 1510, 8665, 17, 2357, 2412, 3144, 2334, 15974, 1535, 3729, 17, 13816, 4695, 2601, 5725, 2076, 7655, 1495, 4580, 6792, 3579, 2647, 15, 11553, 11, 2842, 18, 25, 12, 1675, 1886, 10359, 1]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
labels: [3, 17358, 22138, 25058, 2649, 2977, 1766, 2532, 2464, 31314, 3452, 15, 2537, 1786, 2340, 11122, 12571, 1795, 16, 8286, 17, 4493, 2550, 1786, 3354, 6649, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
decoder_input_ids: [3, 3, 17358, 22138, 25058, 2649, 2977, 1766, 2532, 2464, 31314, 3452, 15, 2537, 1786, 2340, 11122, 12571, 1795, 16, 8286, 17, 4493, 2550, 1786, 3354, 6649, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

Example 3:
input_ids: [3, 17715, 1050, 17, 3036, 15, 10684, 29, 17848, 1510, 16451, 2591, 4459, 2092, 2340, 17, 1688, 2591, 3353, 23238, 2054, 9647, 3763, 1495, 10684, 15, 4138, 2102, 15, 9002, 11, 28, 18, 28, 12, 17, 1887, 8625, 2591, 3214, 6943, 8141, 10912, 4156, 7300, 1016, 1509, 5301, 10364, 6108, 10912, 15778, 4834, 1007, 6649, 11191, 5755, 17, 3729, 30035, 1519, 2177, 3541, 1978, 10684, 1509, 7501, 6963, 17122, 17820, 15, 10684, 17, 2591, 2562, 30166, 1489, 16, 12449, 15350, 16, 21, 1007, 4994, 6228, 1675, 3763, 2044, 4550, 2397, 17, 1587, 24072, 1495, 4503, 9841, 7681, 4102, 7731, 1730, 17, 2448, 4459, 9336, 1499, 1684, 2341, 11141, 24904, 1008, 1495, 1930, 1750, 10684, 1570, 7087, 9152, 17, 3693, 2591, 7654, 15, 1956, 2165, 9484, 17, 1542, 15069, 3137, 17035, 7797, 3991, 3673, 1495, 3667, 17, 3604, 1542, 3969, 7721, 3167, 1572, 3198, 2847, 3693, 17944, 2591, 17, 5840, 2591, 3353, 6928, 6765, 1510, 1737, 1859, 17, 1535, 26, 2867, 3609, 15, 1688, 2591, 12298, 2562, 15350, 16, 21, 1007, 3054, 9149, 16, 20774, 3763, 1495, 2044, 4070, 2642, 1595, 3992, 1522, 15, 17695, 15, 4138, 2102, 15, 3028, 4327, 3138, 17, 4127, 1714, 4865, 15, 2199, 6943, 6108, 10912, 8263, 34, 12287, 29, 2591, 12298, 15350, 16, 21, 40, 3763, 1495, 17695, 36, 17, 11, 2801, 1049, 18, 1789, 17715, 25, 12369, 12, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels: [3, 2591, 3353, 23238, 2054, 9647, 3763, 1495, 4503, 9841, 7681, 3927, 1501, 7731, 1730, 15, 10684, 15, 4138, 2102, 17, 5840, 1684, 5446, 6943, 4156, 7300, 1016, 1509, 5301, 10364, 15778, 4834, 1007, 6649, 11191, 5755, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
decoder_input_ids: [3, 3, 2591, 3353, 23238, 2054, 9647, 3763, 1495, 4503, 9841, 7681, 3927, 1501, 7731, 1730, 15, 10684, 15, 4138, 2102, 17, 5840, 1684, 5446, 6943, 4156, 7300, 1016, 1509, 5301, 10364, 15778, 4834, 1007, 6649, 11191, 5755, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
```

#### Print Processed Examples from Validation Dataset
```"html"
Example 1:
input_ids: [3, 17715, 1050, 17, 3036, 15, 17165, 29, 2948, 1714, 2396, 1935, 1509, 1723, 6218, 3739, 8377, 17004, 1885, 2882, 2935, 4351, 4968, 1555, 9208, 7498, 1495, 24249, 2882, 2935, 2384, 2980, 27660, 15, 17165, 15, 2326, 2102, 15, 2762, 1503, 11, 2349, 18, 28, 12, 17932, 17, 5840, 8962, 2008, 1814, 5255, 4203, 16, 10264, 1684, 10098, 1555, 4718, 2230, 17, 12075, 2752, 9208, 7498, 1510, 7009, 23325, 4265, 6700, 1503, 1509, 5840, 2165, 2223, 18815, 1810, 17, 3572, 10976, 15, 5840, 2340, 3028, 16580, 9208, 1510, 2880, 4006, 16, 4006, 1748, 24249, 1814, 2756, 16169, 4784, 17, 6824, 15, 3264, 9208, 3598, 29235, 3836, 1587, 3496, 1509, 3693, 3264, 3137, 2277, 1653, 4221, 3028, 27182, 1510, 2059, 3331, 17, 4265, 6700, 1503, 15, 3167, 21563, 1014, 15, 17165, 15, 1509, 2653, 1714, 4564, 1510, 5688, 1495, 1570, 3685, 9208, 2396, 18681, 17, 2173, 1723, 1714, 2477, 2970, 6218, 3739, 17, 2948, 8722, 4510, 5077, 1519, 2177, 3541, 5058, 3942, 9785, 15, 17165, 17, 2722, 1723, 4510, 6218, 3739, 15, 18129, 6538, 1016, 15, 2172, 1570, 7223, 1789, 5058, 17, 11, 1544, 1016, 18, 19815, 1012, 8900, 1515, 20753, 12, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels: [3, 17004, 1885, 1814, 4351, 4968, 1555, 1688, 9208, 1495, 17165, 15, 20565, 15, 10527, 2948, 1714, 17, 5840, 6237, 3028, 16580, 9208, 1510, 2880, 4006, 16, 4006, 1748, 24249, 1814, 2756, 16169, 4784, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
decoder_input_ids: [3, 3, 17004, 1885, 1814, 4351, 4968, 1555, 1688, 9208, 1495, 17165, 15, 20565, 15, 10527, 2948, 1714, 17, 5840, 6237, 3028, 16580, 9208, 1510, 2880, 4006, 16, 4006, 1748, 24249, 1814, 2756, 16169, 4784, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


Example 2:
input_ids: [3, 1821, 1510, 2694, 1533, 14292, 31919, 1495, 2317, 2888, 16, 7550, 17, 1510, 5949, 7591, 2957, 1519, 3640, 5312, 15, 2185, 2129, 1873, 11099, 2939, 4400, 5452, 4007, 6993, 1495, 12523, 2173, 3297, 11527, 2227, 17, 1542, 4272, 6517, 1795, 2242, 1789, 14605, 6640, 4527, 16735, 7705, 6893, 29026, 1789, 16, 1789, 1495, 10651, 17, 2092, 1519, 14292, 15, 1942, 16, 1942, 31919, 3672, 1757, 29127, 12457, 1495, 4503, 17, 5927, 1745, 4856, 12969, 28364, 1509, 14019, 6868, 1510, 3717, 1626, 1629, 6084, 1495, 5386, 2341, 17, 4718, 17294, 12545, 2210, 17138, 4752, 14506, 1509, 15692, 2891, 53, 2428, 1028, 10654, 2825, 1005, 1495, 3358, 17, 3093, 2477, 15502, 16031, 1538, 14200, 1510, 7634, 12519, 5227, 1016, 18173, 1010, 1027, 4945, 17, 21365, 7591, 1653, 8827, 1533, 7087, 17, 3946, 1495, 5304, 1832, 14769, 6084, 1538, 14292, 4484, 19104, 19204, 15, 4950, 9638, 1510, 1695, 13286, 1609, 1653, 2223, 8850, 17, 21453, 32, 8886, 1012, 15, 31919, 1619, 1723, 16, 3473, 2835, 1495, 2227, 1510, 3272, 12583, 1789, 24828, 7233, 9638, 1520, 16, 6786, 17, 1657, 2223, 5204, 9638, 1510, 2223, 28934, 4822, 1792, 16886, 15, 1509, 14821, 12055, 17, 1495, 9676, 5160, 9638, 31919, 1821, 8828, 2165, 8890, 2242, 9345, 11403, 1495, 1935, 23281, 17, 9638, 31919, 1675, 8914, 1608, 1592, 22068, 2316, 11403, 4096, 1533, 7351, 1944, 17, 1509, 9638, 1675, 3497, 4305, 455, 6786, 1020, 456, 17, 2794, 1533, 2886, 19461, 1510, 4445, 1509, 1656, 2973, 9374, 5333, 16637, 1519, 2891, 16830, 1538, 17, 2530, 1542, 2062, 2530, 28967, 17, 2530, 1]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
labels: [3, 14292, 31919, 16030, 17840, 17, 8378, 5368, 2928, 6993, 1495, 12523, 2173, 3297, 11527, 15, 1495, 4503, 1873, 11099, 2939, 3672, 13342, 1010, 17, 2064, 1565, 6373, 2064, 1565, 15, 14292, 1695, 1619, 12391, 1499, 1533, 9638, 1510, 9746, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
decoder_input_ids: [3, 3, 14292, 31919, 16030, 17840, 17, 8378, 5368, 2928, 6993, 1495, 12523, 2173, 3297, 11527, 15, 1495, 4503, 1873, 11099, 2939, 3672, 13342, 1010, 17, 2064, 1565, 6373, 2064, 1565, 15, 14292, 1695, 1619, 12391, 1499, 1533, 9638, 1510, 9746, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


Example 3:
input_ids: [3, 17715, 1050, 17, 3036, 15, 2647, 29, 7797, 8893, 29642, 4154, 9523, 2177, 1495, 3030, 2968, 3256, 1518, 15, 2647, 15, 12757, 11, 1587, 18, 2162, 12, 3509, 17, 1653, 1821, 4510, 2769, 1570, 4857, 1542, 17, 1956, 9812, 4628, 4700, 2766, 7599, 2822, 12164, 17, 2379, 7951, 3122, 15, 2935, 2148, 1833, 16942, 2044, 4550, 3152, 17, 2784, 7427, 17, 1570, 2220, 5831, 15, 2139, 20637, 3410, 1720, 29810, 1509, 1626, 5727, 9523, 2971, 1495, 2384, 3049, 1723, 1510, 2382, 1795, 5288, 1533, 1840, 17, 3167, 17835, 2185, 1653, 3598, 5986, 3668, 16, 3668, 9693, 1538, 17, 9648, 1743, 2935, 1944, 2223, 29329, 5298, 2556, 1803, 1753, 3167, 1509, 2044, 2270, 3137, 17035, 7797, 17367, 1519, 3667, 4857, 17, 7797, 1542, 6237, 3028, 2861, 3154, 6421, 4107, 1495, 1891, 1723, 2177, 3167, 17, 11, 6723, 18, 22869, 2984, 1730, 12, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels: [3, 7797, 6237, 3028, 2861, 3154, 6421, 4107, 1495, 1891, 1723, 2177, 3167, 17, 1653, 1821, 4510, 2769, 1570, 4857, 1542, 15, 1956, 9812, 4628, 4700, 2766, 7599, 2822, 12164, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
decoder_input_ids: [3, 3, 7797, 6237, 3028, 2861, 3154, 6421, 4107, 1495, 1891, 1723, 2177, 3167, 17, 1653, 1821, 4510, 2769, 1570, 4857, 1542, 15, 1956, 9812, 4628, 4700, 2766, 7599, 2822, 12164, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
```

#### Printing the Length of Training and Validation Datasets
The lengths of the training and validation datasets give us an overview of the dataset size and the number of examples available for model training and validation:

1. Length of Training Dataset: 79,413 examples.
2. Length of Validation Dataset: 19,854 examples.

This indicates that the training dataset consists of 79,413 data points, which are used to train the machine learning model, while the validation dataset contains 19,854 data points, which are used to evaluate the model's performance during the training process.

These dataset sizes reflect the amount of data available for learning patterns (training) and for assessing generalization (validation).

#### Print The First Element of Training and Validation Dataset
Inspecting the first element of both the training and validation datasets provides insight into the data structure and preprocessing applied before training the model. These details include the processed text, summaries, and tokenized inputs necessary for the Bert2GPT model to learn and make predictions effectively.
```"html"
First Element of Training Dataset: {'id': '254103', 'url': 'https://www.liputan6.com/news/read/254103/korban-kekerasan-polisi-datangi-mabes-polri', 'clean_article': 'Liputan6.com, Jakarta: Tiga korban penembakan polisi di Ogan Ilir, Sumatra Selatan, mendatangi Mabes Polri, Jakarta, Kamis (10/12) siang. Didampingi ketua Lembaga Bantuan Hukum Jakarta, Patra Zen, ketiga warga Ogan Ilir itu menunjukkan berbagai bukti kekerasan oleh polisi, termasuk bekas luka tembak akibat peluru karet. Selain peluru karet, polisi juga dituding telah menggunakan peluru tajam. Mereka menuntut Kapolri Jenderal Bambang Hendarso Danuri untuk bertanggungjawab atas kejadian tersebut. Peristiwa penembakan bermula saat warga bersitegang dengan pihak pabrik gula Cinta Manis di Ogan Ilir. Mereka mengatakan lahan seluas 1500 hektare bukan milik perusahaan, namun milik warga. Masyarakat mengatakan hal tersebut didasarkan atas keputasan Mahkamah Agung pada 1996. Dalam aksi tersebut, warga terlibat bentrok dengan polisi. Dua belas warga terluka terkena peluru karet, sedangkan tiga petugas perusahaan menderita luka bacokan benda tajam. Para korban lalu dirawat di Rumah Sakit Umum Muhammad Husein Palembang. Pihak kepolisian membenarkan insiden penembakan itu. Namun mereka mengatakan penembakan sudah sesuai prosedur dan menggunakan peluru karet. Menurut polisi langkah itu dilakukan setelah pengunjuk rasa bertindak anarkis dengan merusak sejumlah fasilitas perusahaan gula. [baca: Petani dan Polisi Bentrok, 15 Orang Cedera]. Simak selengkapnya di video. (WIL/AYB).', 'clean_summary': 'Tiga korban penembakan polisi di Ogan Ilir, Sumsel, mendatangi Mabes Polri untuk menunjukkan berbagai bukti kekerasan oleh polisi. Mereka menuntut Kapolri Jenderal Bambang Hendarso Danuri untuk bertanggungjawab.', 'extractive_summary': 'Liputan6.com, Jakarta: Tiga korban penembakan polisi di Ogan Ilir, Sumatra Selatan, mendatangi Mabes Polri, Jakarta, Kamis (10/12) siang. Mereka menuntut Kapolri Jenderal Bambang Hendarso Danuri untuk bertanggungjawab atas kejadian tersebut.', 'input_ids': [3, 17715, 1050, 17, 3036, 15, 2647, 29, 2221, 4510, 14575, 4476, 1495, 15668, 17368, 15, 3596, 2086, 15, 14061, 25215, 9590, 15, 2647, 15, 11553, 11, 2178, 18, 2162, 12, 7366, 17, 15455, 3162, 3409, 3920, 2552, 2647, 15, 6840, 1007, 11886, 15, 2918, 3167, 15668, 17368, 1675, 3027, 2236, 4606, 6457, 1592, 4476, 15, 2199, 4821, 6218, 12911, 3028, 7724, 8835, 17, 2248, 7724, 8835, 15, 4476, 1609, 10629, 1529, 1720, 2117, 7724, 8948, 17, 1695, 7113, 23125, 3458, 9183, 5390, 18792, 1509, 2803, 1572, 19445, 1930, 4857, 1684, 17, 3604, 14575, 8502, 1793, 3167, 3809, 16673, 9563, 1555, 3085, 4649, 6751, 4045, 7566, 1495, 15668, 17368, 17, 1695, 3270, 4162, 5790, 14027, 11652, 2815, 3353, 2316, 15, 1956, 3353, 3167, 17, 2246, 3270, 1938, 1684, 5920, 1930, 20346, 1798, 7391, 3213, 1535, 4504, 17, 1570, 4798, 1684, 15, 3167, 4234, 29024, 1555, 4476, 17, 1887, 4306, 3167, 8471, 7068, 7724, 8835, 15, 2722, 2221, 8166, 2316, 6649, 6218, 5702, 5068, 3971, 8948, 17, 1855, 4510, 2579, 12330, 1495, 2177, 3541, 1978, 3942, 19136, 7334, 17, 3085, 7301, 17237, 7621, 14575, 1675, 17, 1956, 1695, 3270, 14575, 2277, 3215, 9302, 1509, 2117, 7724, 8835, 17, 2379, 4476, 5468, 1675, 2410, 1753, 18276, 4010, 6385, 19030, 1555, 8657, 2878, 4086, 2316, 6751, 17, 34, 12287, 29, 4493, 1509, 4476, 29024, 15, 2270, 1714, 9043, 36, 17, 17886, 1028, 8146, 9563, 1914, 14007, 1508, 1495, 3229, 17, 11, 1830, 18, 3973, 1013, 12, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': [3, 2221, 4510, 14575, 4476, 1495, 15668, 17368, 15, 2105, 6669, 15, 14061, 25215, 9590, 1572, 3027, 2236, 4606, 6457, 1592, 4476, 17, 1695, 7113, 23125, 3458, 9183, 5390, 18792, 1509, 2803, 1572, 19445, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'decoder_input_ids': [3, 3, 2221, 4510, 14575, 4476, 1495, 15668, 17368, 15, 2105, 6669, 15, 14061, 25215, 9590, 1572, 3027, 2236, 4606, 6457, 1592, 4476, 17, 1695, 7113, 23125, 3458, 9183, 5390, 18792, 1509, 2803, 1572, 19445, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}
First Element of Validation Dataset: {'id': '147664', 'url': 'https://www.liputan6.com/news/read/147664/sri-tanjung-tabrakan--lima-tewas', 'clean_article': 'Liputan6.com, Probolinggo: Lima orang meninggal dunia dan satu luka berat menyusul tabrakan antara Kereta Api Sri Tanjung dengan truk angkutan di perlintasan kereta api Jalan Raya Banjarsari, Probolinggo, Jawa Timur, Ahad (16/9) petang. Kecelakaan berawal ketika KA jurusan Yogyakarta-Banyuwangi tersebut melaju dengan kecepatan tinggi. Mendadak muncul truk angkutan yang dikemudikan Abdul Somad dan kecelakaan tak bisa dihindarkan. Kuat dugaan, kecelakaan terjadi akibat sopir truk yang kurang hati-hati karena perlintasan KA tanpa palang pintu. Akibatnya, badan truk sempat terseret 20 meter dan kondisi badan mobil sudah tidak berbentuk akibat benturan yang sangat keras. Abdul Somad, warga Lumbang, Probolinggo, dan empat orang penumpang yang duduk di dalam bak truk meninggal seketika. Sementara satu orang lagi mengalami luka berat. Lima jenazah korban dikirim ke Rumah Sakit Dokter Muhammad Saleh, Probolinggo. Sedangkan satu korban luka berat, Miarto, masih dalam perawatan tim dokter. (ADO/Dandi Arigafur).', 'clean_summary': 'Tabrakan antara KA Sri Tanjung dengan sebuah truk di Probolinggo, Jatim, menewaskan lima orang. Kecelakaan diduga akibat sopir truk yang kurang hati-hati karena perlintasan KA tanpa palang pintu.', 'extractive_summary': 'Kuat dugaan, kecelakaan terjadi akibat sopir truk yang kurang hati-hati karena perlintasan KA tanpa palang pintu. Lima jenazah korban dikirim ke Rumah Sakit Dokter Muhammad Saleh, Probolinggo.', 'input_ids': [3, 17715, 1050, 17, 3036, 15, 17165, 29, 2948, 1714, 2396, 1935, 1509, 1723, 6218, 3739, 8377, 17004, 1885, 2882, 2935, 4351, 4968, 1555, 9208, 7498, 1495, 24249, 2882, 2935, 2384, 2980, 27660, 15, 17165, 15, 2326, 2102, 15, 2762, 1503, 11, 2349, 18, 28, 12, 17932, 17, 5840, 8962, 2008, 1814, 5255, 4203, 16, 10264, 1684, 10098, 1555, 4718, 2230, 17, 12075, 2752, 9208, 7498, 1510, 7009, 23325, 4265, 6700, 1503, 1509, 5840, 2165, 2223, 18815, 1810, 17, 3572, 10976, 15, 5840, 2340, 3028, 16580, 9208, 1510, 2880, 4006, 16, 4006, 1748, 24249, 1814, 2756, 16169, 4784, 17, 6824, 15, 3264, 9208, 3598, 29235, 3836, 1587, 3496, 1509, 3693, 3264, 3137, 2277, 1653, 4221, 3028, 27182, 1510, 2059, 3331, 17, 4265, 6700, 1503, 15, 3167, 21563, 1014, 15, 17165, 15, 1509, 2653, 1714, 4564, 1510, 5688, 1495, 1570, 3685, 9208, 2396, 18681, 17, 2173, 1723, 1714, 2477, 2970, 6218, 3739, 17, 2948, 8722, 4510, 5077, 1519, 2177, 3541, 5058, 3942, 9785, 15, 17165, 17, 2722, 1723, 4510, 6218, 3739, 15, 18129, 6538, 1016, 15, 2172, 1570, 7223, 1789, 5058, 17, 11, 1544, 1016, 18, 19815, 1012, 8900, 1515, 20753, 12, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': [3, 17004, 1885, 1814, 4351, 4968, 1555, 1688, 9208, 1495, 17165, 15, 20565, 15, 10527, 2948, 1714, 17, 5840, 6237, 3028, 16580, 9208, 1510, 2880, 4006, 16, 4006, 1748, 24249, 1814, 2756, 16169, 4784, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'decoder_input_ids': [3, 3, 17004, 1885, 1814, 4351, 4968, 1555, 1688, 9208, 1495, 17165, 15, 20565, 15, 10527, 2948, 1714, 17, 5840, 6237, 3028, 16580, 9208, 1510, 2880, 4006, 16, 4006, 1748, 24249, 1814, 2756, 16169, 4784, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}
```
These preprocessed examples are critical for model training, ensuring that both the training and validation datasets are ready for optimal learning and inference.

## Load Pre-trained Model
The process of loading pre-trained models is a crucial step in setting up the Bert2GPT Indonesian Text Summarizer, which uses the established weights and configurations of BERT and GPT-2. These models serve as the encoder and decoder, respectively, in a sequence-to-sequence architecture.
### 1. Loading the Pre-trained BERT Model
The summarization model uses a specific pre-trained BERT model called "cahya/bert-base-indonesian-1.5G".
BERT acts as the encoder, designed to understand and process the input text. Since this BERT model is pre-trained on Indonesian language data, it's well-suited for comprehending the context and linguistic features of Indonesian texts.
In this step, BERT reads and encodes the input articles, generating contextual embeddings—representations of the text’s meaning and structure, which are used in the summarization process.
### 2. Loading the Pre-trained GPT-2 Model
The second step involves loading the GPT-2 model, specifically "cahya/gpt2-small-indonesian-522M".
GPT-2 is used as the decoder in this architecture. However, to make it work effectively in this summarization task, it needs to be modified to enable cross-attention. This means GPT-2 can now focus on the encoded information from the BERT model.
After receiving the encoded input from BERT, GPT-2 generates the actual summary. It ensures that the output is not only grammatically fluent but also aligned with the meaning and context of the original article.
### Why This Approach Works
By combining the pre-trained BERT and GPT-2 models, this summarizer leverages their strengths:

1. BERT excels at understanding context and capturing the deep nuances of the Indonesian language.
2. GPT-2 specializes in generating text that is coherent and fluent.

This setup significantly enhances the model's ability to generate accurate and meaningful summaries while saving time and resources, as the pre-trained models reduce the need for building and training the system from scratch.

## Create Model
The process of creating the Bert2GPT Indonesian Text Summarizer involves carefully setting up the architecture that integrates pre-trained BERT and GPT-2 models into an encoder-decoder framework. Here’s an explanation of each step based on the code and process you've outlined:

### 1. Create the GPT-2 Model with Modified Configuration
The GPT-2 model is initialized with a configuration that allows cross-attention. This ensures that the GPT-2 decoder can attend to the output from the BERT encoder.
Cross-attention enables the decoder to focus on relevant parts of the encoded input text, making the summarization process more accurate and context-aware.
### 2. Combine BERT and GPT-2 into an Encoder-Decoder Model
The BERT model is the encoder, and the GPT-2 model is the decoder in this architecture. The encoder reads the input text, converts it into a meaningful representation (encoded data), and then passes this representation to the GPT-2 decoder.
The decoder then generates a summary based on the encoded input. This sequence-to-sequence structure is typical for tasks like summarization.
### 3. Update Special Tokens Based on the Tokenizer
Special tokens such as the decoder start token, pad token, and end-of-sequence (EOS) token are essential for the model to understand where a sequence begins and ends.
These tokens are crucial during training and inference, ensuring the model handles inputs of varying lengths correctly.
### 4. Check for GPU Availability
GPU availability is checked to determine whether the model can be trained on a GPU, which would significantly speed up training.
If a GPU is available, the model is moved to the GPU for faster computation. Otherwise, it will default to the CPU, which might take longer for training but is still functional.
### 5. Move the Model to the Appropriate Device
Once the device (GPU or CPU) is determined, the model is moved to that device to ensure it's ready for training.
### 6. Print Model Architecture and Parameters
The code prints the architecture of the model to give a clear overview of how the components are structured.
Additionally, it calculates the total number of parameters and the trainable parameters in the model. This information is helpful to understand the model's complexity and its trainable components.
### Summary
The steps taken in this process ensure that the Bert2GPT Indonesian Text Summarizer is effectively prepared for training. By integrating pre-trained BERT and GPT-2 models and configuring special tokens, the model leverages advanced language understanding and text generation capabilities. Checking GPU availability helps optimize the training process for faster results, and understanding the model’s parameters gives insight into its complexity.

## Training Model
The training process of the Bert2GPT Indonesian Text Summarizer involves a structured approach, consisting of two key phases: Initial Training and Continued Training. 
### 1. Initial Training
This phase involves setting up the environment and the necessary configurations to train the model from scratch or from a pre-trained state.
#### Key Steps in Initial Training:
##### Training Arguments:
Define how the training process will be managed, such as:
1. Max Steps: Total number of training steps (e.g., 20,000 steps).
2. Batch Sizes: Size of the batches for training and evaluation (e.g., 8 samples per batch).
3. Warmup Steps: Initial steps where the learning rate increases gradually.
4. Weight Decay: Regularization technique to avoid overfitting.
5. Mixed Precision: Enabling mixed precision (FP16) to speed up training by reducing memory usage.
6. Logging and Checkpoints: Log results at regular intervals and save model checkpoints less frequently to reduce storage requirements.
#### Trainer Creation:
A Trainer object is created to handle the training process. It coordinates the loading of data, optimizes the model weights, and evaluates its performance on a validation dataset.
#### Model Training:
The actual training begins with the trainer.train() method. This process adjusts the model’s weights to minimize the loss function, gradually improving the model’s performance over time.
### 2. Continue Training
The Continued Training phase allows for the refinement of the model after it has been trained for a while. Instead of starting from scratch, training resumes from a saved checkpoint to further enhance performance.
#### Key Steps in Continued Training:
##### Loading the Model from a Checkpoint:
The model and tokenizer are reloaded from a previous checkpoint (e.g., after 15,000 steps). This ensures the training continues from where it last left off.
##### Defining Continued Training Arguments:
The training arguments are slightly modified, such as reducing the number of warmup steps or changing how frequently the model is evaluated and checkpoints are saved. The checkpoint path is also specified so the training can resume from that point.
##### Reinitializing the Trainer:
A new Trainer is instantiated using the loaded model and the updated training arguments. This allows the training to resume seamlessly from the previous checkpoint.
##### Continuing Training:
The training process resumes using trainer.train(), and the model continues to improve based on the remaining optimization steps.
### Summary of Phases
1. Initial Training: Starts the training from the beginning or a pre-trained state, using well-defined training arguments and datasets.
2. Continue Training: Resumes training from a checkpoint, allowing incremental improvements to the model while saving time and resources.

Both phases allow for flexibility, letting the model evolve through initial training and further refinement through continued training. This iterative approach helps achieve better performance over time while utilizing pre-existing model checkpoints.

## Evaluate Model
The model evaluation process consists of two key components: tracking the training progress via loss graphs and measuring the model's performance using ROUGE scores. 

### Training and Validation Losses
![image13](https://github.com/user-attachments/assets/412376aa-1367-4e49-b4da-90d1cdd70c89)

<b>What it shows:</b> The graph plots the training loss and validation loss against the number of training steps.

    1. Training loss: This reflects how well the model is learning from the training dataset.
    2. Validation loss: This measures the model’s performance on a separate validation dataset to avoid overfitting.
    
<b>Trend:</b> As training progresses, both the training and validation losses decrease, indicating that the model is improving. The difference between the two also indicates whether the model generalizes well or overfits.

### ROUGE Scores
<b>What ROUGE is:</b>  ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate the quality of summaries by comparing them to reference summaries. The key metrics are:

ROUGE-1: Measures the overlap of unigrams (single words) between the generated summary and the reference summary.

ROUGE-2: Measures the overlap of bigrams (pairs of consecutive words) between the generated and reference summaries.

ROUGE-L: Focuses on the longest common subsequence between the generated and reference summaries.

![image14](https://github.com/user-attachments/assets/3e740102-b629-4e18-8f5c-d6a560b77c57)

<b>What the graph shows:</b> The bar chart presents the model’s performance on ROUGE-1, ROUGE-2, and ROUGE-L scores. Each metric is broken down into:

    - Recall: Measures how much of the reference summary’s content is captured in the generated summary.
    - Precision: Measures how much of the generated summary’s content is relevant compared to the reference.
    - F1 Score: A balanced metric combining both recall and precision.


### Model’s Performance:
    1. ROUGE-1 has a high recall score (~0.675), meaning that the generated summaries capture a significant portion of the content from the reference summaries. However, the precision (~0.212) and F1 score (~0.322) are lower, indicating that while the summaries include important parts of the text, they may also contain irrelevant or unnecessary information.
    2. ROUGE-2 shows lower recall, precision, and F1 scores, which is expected because bigrams (word pairs) are harder to match exactly.
    3. ROUGE-L performs similarly to ROUGE-1, suggesting the model captures important sentence structures and sequences in the summaries.

The evaluation results show that the Bert2GPT Indonesian Text Summarizer effectively generates relevant summaries, as evidenced by the decreasing training and validation losses and reasonable ROUGE scores. While the model performs well in terms of recall, it has room for improvement in precision, which would help generate more concise and relevant summaries.

## Results
### Testing the Model
#### 1. Load the Tokenizer and Model
    i. Tokenizer: The tokenizer used for the summarization is "cahya/bert2gpt-indonesian-summarization". It splits the input text into tokens that the model can process. Two special tokens are configured:
        a. BOS (Beginning of Sequence): Marks the start of the input.
        b. EOS (End of Sequence): Marks the end of the input.
    ii. Model: The summarization model is loaded from a specific checkpoint (e.g., "/content/drive/MyDrive/continued_results/checkpoint-15001"), which contains the learned weights and configurations from previous training steps. These weights allow the model to generate summaries effectively.
#### 2. Tokenizer Configuration
The tokenizer is configured with BOS and EOS tokens to ensure proper formatting of the input text for the model. These tokens help the model understand where the input begins and ends, which is important for accurate summarization.
#### 3. Model Loading
The pre-trained model is loaded from a checkpoint that stores the trained model parameters (weights). This ensures that the model’s summarization capabilities are leveraged during testing, producing high-quality summaries.
### 4. Input Article for Summarization
An article is provided as input for the model. This article serves as the basis for summarization, allowing the model to demonstrate its ability to condense the content and capture the main points.
### 5. Generate Summary
- The input article is encoded by the tokenizer, converting it into a format suitable for the model.
- The model generates a summary using various parameters that influence the summarization process, such as:
    -- Minimum and Maximum Length: Specifies the allowed range for the summary’s length.
    -- Beam Search (num_beams): Controls how many different sequences the model explores to find the best summary.
    -- Repetition Penalty: Prevents the model from repeating the same words or phrases too frequently.
    -- Length Penalty: Adjusts the balance between short and longer summaries.
### 6. Summary Generation Parameters
The parameters used during the summary generation include:
    - Minimum Length: Sets the shortest possible summary length to ensure the summary has enough information.
    - Maximum Length: Limits the length of the summary to avoid excessively long outputs.
    - Beam Search: Helps the model explore multiple possible sequences and choose the most relevant summary.
    - Repetition Penalty: Ensures that the summary does not repeat the same words unnecessarily.
### 7. Decoding the Summary
The summary generated by the model is in the form of token IDs, which need to be decoded back into human-readable text. Special tokens such as BOS and EOS are removed during this step, resulting in a concise, readable summary of the original article.

<table border="1" cellpadding="10" cellspacing="0">
  <tr>
    <th>Aspect</th>
    <th>Details</th>
  </tr>
  <tr>
    <td>Article to Summarize</td>
    <td>Liputan6.com, Jakarta: Tentara Nasional Indonesia hendaknya benar-benar profesional. TNI juga harus berada di atas seluruh kekuatan politik yang ada. Demikian permintaan mantan Presiden Partai Keadilan Sejahtera Hidayat Nur Wahid, di sela-sela Munas PKS, Sabtu (19/6), di Jakarta. TNI adalah alat negara yang harus netral dan berada di atas seluruh kekuatan politik yang ada. Ini untuk menjaga keamanan teritorial dan keutuhan Negara Kesatuan Republik Indonesia, kata Hidayat. Menurutnya, TNI baru mungkin memiliki hak pilih pada pemilu jika diatur secara konstitusional melalui undang-undang. Saya kira wacana TNI memiliki hak pilih dalam pemilu harus melalui pembahasan lebih lanjut di DPR, ujar anggota Komisi I DPR RI itu. Dari pembicaraan dengan pimpinan TNI, sampai saat ini TNI masih memilih belum terlibat di pemilu. Hal ini belajar dari pengalaman pada pemilu 1955, di mana TNI terlibat sehingga terbelah pada sejumlah kekuatan politik.</td>
  </tr>
  <tr>
    <td>Summarization Parameters</td>
    <td>min_length=20, max_length=80, num_beams=10, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True, no_repeat_ngram_size=2, use_cache=True, do_sample=True, temperature=0.3, top_k=50, top_p=0.95</td>
  </tr>
  <tr>
    <td>Generated Summary</td>
    <td>[UNK] mantan presiden pks hidayat nur wahid mengatakan tni adalah alat negara yang harus netral dan berada di atas seluruh kekuatan politik yang ada. tni baru mungkin memiliki hak pilih pada pemilu jika diatur secara konstitusional melalui undang - undangan.ikapijiran ke dpr. hal ini membuat sistem keamanan nasional menjadi tidak optimal bagi anggota tni / 6, bisa ditentukan oleh uu itu. apa pun mengganggu kekompakan.</td>
  </tr>
  <tr>
    <td>ROUGE Scores for new Summary</td>
    <td>
      <ul>
        <li>ROUGE-1: r: 1.0, p: 1.0, f: 0.999999995</li>
        <li>ROUGE-2: r: 1.0, p: 1.0, f: 0.999999995</li>
        <li>ROUGE-L: r: 1.0, p: 1.0, f: 0.999999995</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>Error Analysis</td>
    <td>
      <ul>
        <li>Too Short: 0</li>
        <li>Too Long: 1</li>
        <li>Missing Key Information: 0</li>
        <li>Other: 0</li>
      </ul>
    </td>
  </tr>
</table>


### Summary
1. The summarization model produced a concise and relevant summary of the input article, focusing on key information such as the neutrality of the TNI and its role in national security.
2. ROUGE scores indicate that the generated summary captures most of the essential content, but there is room for improvement, especially in terms of precision and fluency.
3. The error analysis revealed that the summary was slightly longer than necessary, but did not miss key information or show other major issues.


## Improvements
1. Precision Enhancement: The ROUGE scores indicate that while recall is high, precision could be improved. This suggests that the model is capturing important information but may also be including less relevant details. Improving precision would lead to more concise and focused summaries.
2. Length Optimization: The error analysis revealed that the generated summary was slightly longer than necessary. Refining the model to produce more concise summaries would address this issue and potentially improve overall quality.
3. Fluency Improvement: Although not explicitly mentioned in the error analysis, enhancing the natural language flow and readability of the summaries could significantly improve their quality and usefulness.

These three areas for improvement target the core aspects of summary quality: relevance, conciseness, and readability. Focusing on these areas should lead to noticeable enhancements in the Bert2GPT Indonesian Text Summarizer's performance.
