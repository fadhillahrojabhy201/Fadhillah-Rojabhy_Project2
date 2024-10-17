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
Similarly, we analyze the distribution of summary lengths to understand the patterns in the summarization process. This step is essential to ensure that the generated summaries are concise while still retaining key information. A histogram is used to visualize how summary lengths vary, aiding in fine-tuning the model to produce balanced and informative summaries.
![image2](https://github.com/user-attachments/assets/1d920f98-821c-4753-8b07-8dab6ef77399)
The histogram shows a relatively normal distribution, with a peak around 170-210 characters, indicating that most summaries are concise and fall within a tight range. The curve tails off sharply after 300 characters, indicating that very long summaries are rare.
The dataset is consistent in terms of summary lengths, with most summaries being short and informative. The distribution's tight range suggests that the model will focus on generating concise summaries around 190 characters, with minimal outliers or extreme summary lengths. This information is important for training the model to generate summaries that are concise, aligned with the dataset, and relevant.




