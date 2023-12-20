# Neural-Text-Generation-Character-to-Word-Level-LSTM-Models
Part A: Developed and fine-tuned a character-level model to recognize Java syntax, avoiding overfitting while ensuring generalization.

Part B: Created a generative model using PyTorch, LSTM networks, GloVe embeddings, and beam search. Trained on Jane Austen's "Persuasion" for coherent sentence generation. Experimented with hyperparameters for realistic output.

## Description

In Part A, the primary focus was on developing a character-level model capable of recognizing and understanding Java syntax. The critical goal was to ensure that the model didn't merely memorize the provided Java code but could generalize effectively to recognize typical Java programming patterns. Achieving this balance between memorization and comprehension required careful consideration of hyperparameters and extensive experimentation. By thoroughly reviewing and understanding the provided code, conducting a literature review, and preparing the dataset, I set the foundation for tackling this complex task.

In Part B, I transitioned to a different domain, where the aim was to create a generative model. Leveraging PyTorch, LSTM networks, GloVe embeddings, and beam search, I trained the model on Jane Austen's novel "Persuasion." The challenge here was to generate coherent and novel sentences while emulating the writing style of the renowned author. This endeavor involved an intricate process of hyperparameter tuning, where factors like sample length, network architecture, dropout, and temperature of the softmax function played pivotal roles in achieving the desired balance between creativity and adherence to the author's style. Additionally, I implemented beam search for sentence generation and introduced a goodness function to evaluate the quality of generated text.

In conclusion, this project not only demonstrates the versatility of machine learning models but also highlights their adaptability across different domains and problem types. The insights gained from both parts of this project contribute to the broader field of natural language processing and open up possibilities for a wide range of applications, from code understanding to creative text generation.

## Getting Started

### Dependencies
- **Python Libraries:** 
  - `math`
  - `numpy`
  - `tqdm`
  - `collections` from `collections`
  - `Counter` from `collections`
  - `pandas`
  - `re`
  - `matplotlib.pyplot` as `plt`
  - `torch` (PyTorch library)
  - `sklearn` (scikit-learn library)
  - `nltk` (Natural Language Toolkit)
  - `keras.preprocessing.text` (from Keras library)
  - `keras.preprocessing.sequence` (from Keras library)
  
- **PyTorch Dependencies:** 
  - `torch.utils.data.Dataset`
  - `torch.utils.data.DataLoader`
  - `torch.nn.functional` as `F`
  - `torch.utils.data.random_split`

- **Scikit-Learn Dependencies:**
  - `PCA` from `sklearn.decomposition`
  - `TruncatedSVD` from `sklearn.decomposition`
  - `TfidfVectorizer` from `sklearn.feature_extraction.text`
  - `CountVectorizer` from `sklearn.feature_extraction.text`

- **Keras Dependencies:**
  - `Tokenizer` from `keras.preprocessing.text`
  - `pad_sequences` from `keras.preprocessing.sequence`

These dependencies were used in various parts of the project for data preprocessing, model development, and evaluation. Make sure to have these libraries installed to run the code successfully.

### Installing

* Please download all the required libraries and datasets 
* Make sure to keep everything in the same directory or Reroute properly

## Authors

contact info

Mohan Thota 
mohant@bu.edu/mohan5thota@gmail.com

## License

This project is licensed under the [APACHE 2.0] License - see the LICENSE.md file for details

## Acknowledgments

Guidance, code snippets, etc.
* [Proffessor](https://www.bu.edu/cs/profiles/wayne-snyder/)
ha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
