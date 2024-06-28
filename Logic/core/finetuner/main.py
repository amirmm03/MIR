from BertFinetuner_mask import BERTFinetuner
from huggingface_hub import login
login('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')

# Instantiate the class
bert_finetuner = BERTFinetuner('IMDB_crawled.json', top_n_genres=5)

# Load the dataset
bert_finetuner.load_dataset()

# Preprocess genre distribution
bert_finetuner.preprocess_genre_distribution()

# Split the dataset
bert_finetuner.split_dataset()

# Fine-tune BERT model
bert_finetuner.fine_tune_bert(0.1,0.1)

# Compute metrics
bert_finetuner.evaluate_model()

# Save the model (optional)
bert_finetuner.save_model('Movie_Genre_Classifier')

#                   precision    recall  f1-score   support

#            Drama       0.82      0.87      0.84        67
# Action/Adventure       0.68      0.78      0.72        32
#           Comedy       0.57      0.18      0.28        22
#            Crime       0.80      0.24      0.36        17
#   Sci-Fi/Fantasy       0.72      0.68      0.70        19

#        micro avg       0.75      0.66      0.71       157
#        macro avg       0.72      0.55      0.58       157
#     weighted avg       0.74      0.66      0.67       157
#      samples avg       0.74      0.66      0.67       157