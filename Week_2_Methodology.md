# NLP Feedback Intelligence — Week 2 Methodology

## 1. Introduction and Project Context

This document defines the data preparation strategy for NLP Feedback Intelligence, a proposed prototype that analyzes written customer feedback. The project focuses on converting unstructured product reviews, survey comments, and support messages into consistent representations suitable for exploratory analysis and sentiment modelling. Poor-quality, duplicated, noisy, or incorrectly labelled data can reduce model reliability. This is a methodology plan; implementation and measured results will be documented in later stages.

## 2. Data Collection Strategy

The dataset will be selected from a reputable public source, subject to its licence and permitted research use. Candidate sources include documented product-review datasets on Kaggle or the Hugging Face Datasets Hub. Selection criteria include source, licence, language, fields, label definitions, size, and documentation. If a suitable dataset cannot be used, a hypothetical anonymized feedback dataset may be used to demonstrate the workflow.

Planned fields include record_id, review_text, rating or sentiment_label, product_category, and date where available. Personal identifiers such as names, email addresses, phone numbers, and addresses will not be collected or published.

Collection steps: identify candidate datasets; inspect documentation and licence; confirm relevance and label meaning; download through the permitted source; preserve an untouched raw copy; record source, access date, version, schema, and filtering decisions. Avoid scraping sites where permission or terms do not allow it.

## 3. Data Inspection and Quality Assessment

Inspect row and column counts, data types, missing values, duplicates, text-length distribution, language consistency, and label frequencies. Manually review a sample for HTML fragments, URLs, emojis, repeated punctuation, spelling variation, and boilerplate. Check whether labels come from star ratings or human annotation. A possible mapping is 1–2 stars as negative, 3 as neutral, and 4–5 as positive, but only if the dataset's definitions justify it.

## 4. Text Preprocessing Methodology

Apply the same documented transformations consistently. First identify null or empty review_text values and report exclusions. Detect exact duplicates and duplicate IDs; inspect near-duplicates rather than deleting them automatically. Preserve original text and create a separate cleaned field. Normalize Unicode and whitespace, and lowercase for the baseline where appropriate. Remove HTML tags and replace irrelevant URLs with a placeholder rather than deleting surrounding words. Decide carefully how to handle punctuation, numbers, and emojis because they may carry sentiment or domain meaning.

Tokenization splits text into units using a suitable tokenizer such as scikit-learn, NLTK, or spaCy. Remove stop words only when they do not add meaning; retain negations such as “not”, “no”, and “never”. Stemming reduces words to rough roots, while lemmatization maps words to dictionary base forms. Lemmatization may improve readability; stemming may be faster. The final choice should be tested. Confirm language scope and document mixed-language records. Split data before fitting vocabulary or TF-IDF weights so test data cannot leak into training.

## 5. Feature Engineering and Feature Extraction

Bag of Words represents documents by word counts. It is simple but ignores word order and may emphasize common terms. TF-IDF weights terms that are frequent in a document but less common across the corpus. It is a lightweight and interpretable baseline. Word n-grams capture sequences: unigrams represent single words and bigrams capture phrases such as “not good” or “very helpful”.

Word embeddings such as Word2Vec and GloVe represent words as dense vectors and can capture semantic similarity, but need suitable pretrained vectors or sufficient data and a method to combine word vectors into document vectors. Contextual embeddings from transformer models such as BERT account for surrounding context but require more compute and careful model selection.

The proposed primary representation is TF-IDF with unigrams and bigrams because it is efficient, explainable, reproducible, and compatible with Logistic Regression or Multinomial Naive Bayes. Embeddings may be compared later if resources and dataset size justify them.

## 6. Illustrative Pseudocode

Planned pseudocode (not a claim that the complete pipeline has already run):
1. Load the permitted dataset.
2. Validate required columns and inspect missing values.
3. Preserve raw_review_text and create cleaned_text.
4. Remove unusable rows and justified exact duplicates.
5. Normalize whitespace and selected HTML/URL noise.
6. Split into training and test sets, stratifying by label when possible.
7. Fit TF-IDF on training text only with ngram_range=(1, 2).
8. Transform training and test text with the fitted vectorizer.
9. Train a baseline classifier.
10. Evaluate using accuracy, precision, recall, F1-score, and a confusion matrix.

Python-style outline:
    df = load_dataset()
    df = df.dropna(subset=['review_text'])
    df['clean_text'] = df['review_text'].apply(clean_text)
    train_text, test_text, y_train, y_test = stratified_split(df)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate(y_test, predictions)

## 7. Potential Pitfalls and Mitigation

Privacy: use public or anonymized data and do not publish personal records. Licence: verify permitted use and record provenance. Class imbalance: inspect label counts, use stratified splitting, and consider macro-F1. Negation loss: retain words such as “not” and “never”. Over-cleaning: preserve meaningful punctuation, emojis, numbers, and domain terms. Duplicate leakage: check duplicates before splitting. Data leakage: fit vectorizers only on training data. Label noise: document label creation and inspect samples. Language variation: define supported languages and document slang or code-mixing limitations. Reproducibility: record package versions, random seeds, parameters, and dataset version.

## 8. Tools and Technologies

Python is the primary language. pandas and NumPy support data inspection; regular expressions handle selected text patterns; NLTK or spaCy support tokenization and linguistic processing; scikit-learn supports splitting, TF-IDF, baseline models, and metrics; Matplotlib or Seaborn support plots. Jupyter Notebook or Google Colab can support experimentation, and GitHub will store versioned code and documentation. Final choices depend on the dataset and environment.

## 9. Deliverables and Quality Checks

Planned deliverables include a documented dataset source and schema, data-quality summary, reproducible preprocessing pipeline, comparison of BoW/TF-IDF/embeddings, pseudocode or implementation examples, and a record of limitations. Quality checks will confirm that cleaned text is not empty, labels are valid, duplicate handling is documented, train/test separation is respected, and preprocessing is consistent.

## 10. Conclusion

This methodology establishes a careful and reproducible approach to preparing customer feedback for NLP analysis. It prioritizes provenance, quality checks, privacy, meaningful normalization, and leakage-free feature extraction. TF-IDF with unigrams and bigrams is the practical baseline; word embeddings and transformer representations remain optional extensions. The prepared data and feature pipeline will support later modelling and evaluation.

