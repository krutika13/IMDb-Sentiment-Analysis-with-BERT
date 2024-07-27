# IMDb-Sentiment-Analysis-with-BERT
This project demonstrates how to perform sentiment analysis on IMDb movie reviews using BERT (bert-base-uncased). The dataset is loaded, tokenized, and used to train a BERT model for classifying reviews as positive or negative.
#Data Preparation
Here's a shorter version of the dataset section:

---

## Dataset

The IMDb dataset consists of 50,000 reviews, split into 25,000 training and 25,000 test samples, with an equal number of positive and negative reviews. The training and test sets are from disjoint movie sets to prevent overfitting. Negative reviews are scored ≤4, and positive reviews are scored ≥7. Additionally, 50,000 unlabeled reviews are provided for unsupervised learning. For this project, we use a subset of 10,000 samples.

---

##Fine-Tuning

The BERT (bert-base-uncased) model is fine-tuned for 2 epochs on the IMDb dataset. We use the AdamW optimizer with weight decay and a learning rate of 1e-5, and adjust the learning rate with get_linear_schedule_with_warmup.

After each epoch, the model is evaluated on the validation set using metrics like loss, accuracy, precision, recall, and F1 score. Training stops early if validation loss does not improve for 2 consecutive epochs. The best model is saved as final_fine_tuned_bert_semantic_model.

---

##Overfitting Handling

1. Early Stopping: Training halts if validation loss does not improve for 2 consecutive epochs.
2. Validation Set: Used to monitor model performance and detect overfitting.
3. Weight Decay: Applied through the AdamW optimizer to regularize the model and prevent large weights.
   
---


## Conclusion

This project demonstrates effective sentiment analysis using the BERT model (`bert-base-uncased`). By fine-tuning the pre-trained model on the IMDb dataset and employing strategies such as early stopping, validation monitoring, and weight decay, we ensure robust performance while mitigating overfitting. The result is a well-tuned model capable of accurately classifying movie reviews as positive or negative.

---
