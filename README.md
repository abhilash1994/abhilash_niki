# Niki.ai project

Project to classify questions in set categories like "What", "When", "Why", "Affirmation" and "Unknown"

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
pip install -r requirements.txt
```

### Running the code

Clone the directory

```
git clone https://github.com/abhilash1994/abhilash_niki.git
```
run the main.py file in the diretory cloned

```
python main.py
```

### Machine learning technique used

* TF-IDF vectorizer
  * Used a ngram of range 1-3.
  * Standard tokenizer of NLTK used to tokenize words.
  * 'min_df' = 0.001 after hyper parameter tuning.
* SVM Stochastic Gradient descent Classifier

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
