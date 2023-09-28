import pandas as pd
import torch
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
train_df = pd.read_csv(r"C:\Users\Ameyo\OneDrive\Desktop\Hinglish-TOP-Dataset-main\Hinglish-TOP-Dataset-main\Dataset\Human Annotated Data\train.tsv", delimiter='\t')
valid_df = pd.read_csv(r"C:\Users\Ameyo\OneDrive\Desktop\Hinglish-TOP-Dataset-main\Hinglish-TOP-Dataset-main\Dataset\Human Annotated Data\validation.tsv", delimiter='\t')
test_df = pd.read_csv(r"C:\Users\Ameyo\OneDrive\Desktop\Hinglish-TOP-Dataset-main\Hinglish-TOP-Dataset-main\Dataset\Human Annotated Data\train.tsv", delimiter='\t')

model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def preprocess_dataset(df, max_seq_length):
    dataset = []
    for index, row in df.iterrows():
        en_query = row['en_query']
        cs_query= row['cs_query']
        input_ids = tokenizer.encode(en_query, return_tensors="pt", padding="max_length", max_length=max_seq_length, truncation=True)
        target_ids = tokenizer.encode(cs_query, return_tensors="pt", padding="max_length", max_length=max_seq_length, truncation=True)
        data_item = {
            'input_ids': input_ids[0],
            'labels': target_ids[0],
        }
        dataset.append(data_item)
max_sequence_length = 128  

train_dataset = preprocess_dataset(train_df, max_sequence_length)
valid_dataset = preprocess_dataset(valid_df, max_sequence_length)
test_dataset = preprocess_dataset(test_df, max_sequence_length)

training_args = Seq2SeqTrainingArguments(
    output_dir=r"C:\Users\Ameyo\OneDrive\Desktop\Hinglish-TOP-Dataset-main\Hinglish-TOP-Dataset-main",
    per_device_train_batch_size=10,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,  
    logging_steps=10,  
    save_steps=500,  
    num_train_epochs=2,  
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,  
)
trainer.train()
trainer.save_model() 
test_results = trainer.predict(test_dataset)
from sacrebleu import corpus_bleu
predicted_translations = [tokenizer.decode(ids, skip_special_tokens=True) for ids in test_results.predictions]
reference_translations = test_df['cs_query'].tolist()

bleu = corpus_bleu(predicted_translations, [reference_translations])
print(f"BLEU score on the test dataset: {bleu.score}")
