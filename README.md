Hinglish Translation with MarianMT

Overview

This Python script demonstrates how to fine-tune the MarianMT model for Hinglish (a blend of Hindi and English) translation using the Hugging Face Transformers library. It also includes a basic translation app for real-time user input translation. It uses fine tuned model specifically trained on hinglish-TOP for better output.

Requirements:-

Python 3.x

Pandas

PyTorch

Hugging Face Transformers Library

Hinglish-TOP dataset

Usage:-

Install the required Python packages:

'pip install pandas torch transformers sacrebleu'

Prepare your training, validation, and test datasets in TSV format (tab-separated values) with columns 'en_query' and 'cs_query'.

Set the model_name variable to the MarianMT model you want to fine-tune. Make sure you have access to the model on the Hugging Face Model Hub.

Adjust the max_sequence_length and training hyperparameters (per_device_train_batch_size, eval_steps, logging_steps, save_steps, num_train_epochs, etc.) as needed for your specific use case.

Run the script to fine-tune the model:

'python fine_tune_hinglish_translation.py'

The script will save the fine-tuned model in the specified output_dir.

You can evaluate the model's performance on the test dataset using the BLEU score.

Translation App:-
The code includes a simple translation app that takes an English sentence as input and translates it to Hinglish. To use the app:

Run the script.

When prompted, enter an English sentence for translation.

The app will provide the Hinglish translation as output.

Type 'exit' to quit the app.

Note:-

Experiment with different hyperparameters to achieve the desired translation quality and training efficiency.





