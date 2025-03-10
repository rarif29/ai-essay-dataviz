# ai-essay-dataviz
AI vs. Human-Written School Essays Analysis

Project Overview:
This project analyzes patterns in word usage and topic differences between AI-generated and human-written school essays. The dataset consists of essays labeled as either AI-generated (coded as 1) or human-written (coded as 0). The goal is to identify lexical features, frequent phrases, and thematic tendencies of each category using text preprocessing and topic modeling techniques.

key components of the analysis include:
- Text Preprocessing: Using spaCy and NLTK for tokenization, lemmatization, and dependency parsing.
- Lexical Analysis: Extracting frequent words and multi-word expressions using n-grams.
- Topic Modeling: Implementing LDA (Latent Dirichlet Allocation) and enhancing it with keyword-based topic mapping to ensure meaningful categorization.
- Visualization: Creating bar graphs, word clouds, treemaps, and other visual aids to illustrate differences between AI and human writing.

Installation Instructions:
To run this project, install the required Python libraries:

pip install pandas spacy nltk scikit-learn matplotlib seaborn wordcloud
python -m spacy download en_core_web_sm


Usage Guide:
1. Load the Dataset: The dataset should be in CSV format, with a column for text and a corresponding label (0 for human-written, 1 for AI-generated).

2. Preprocess the Text:
   - Convert text to lowercase.
   - Remove special characters and stop words.
   - Perform lemmatization using spaCy.

3. Feature Extraction:
   - Generate term frequency-inverse document frequency (TF-IDF) matrices.
   - Extract n-grams (bigrams and trigrams) to analyze common phrase patterns.

4. Topic Modeling:
   - Fit an LDA model with a predefined number of topics.
   - Use keyword-based mapping to refine topic assignments where LDA misclassifies themes.

5. Visualization (up to you, but this is what I did):
   - Generate word clouds for each category.
   - Use treemaps to represent topic distributions.
   - Analyze topic proportions between AI and human-written essays.

Dataset Credit:
Kaggle: Augmented data for LLM - Detect AI Generated Text by Jonathan Hererra. (https://www.kaggle.com/datasets/jdragonxherrera/augmented-data-for-llm-detect-ai-generated-text)
Note: I halved the dataset (80k observations to 40k) for more efficient processing.
