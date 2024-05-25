import os
# Install necessary libraries and packages
os.system("pip install --upgrade pip")
os.system("pip install accelerate -U")
os.system("pip install datasets>=2.6.1")
os.system("pip install git+https://github.com/huggingface/transformers")
os.system("pip install librosa")
os.system("pip install evaluate>=0.30")
os.system("pip install jiwer")
os.system("pip install gradio")
os.system("pip install --upgrade datasets transformers accelerate soundfile librosa evaluate jiwer tensorboard gradio")
os.system("pip install transformers[torch]")
os.system("pip install accelerate>=0.21.0")
os.system("pip install webvtt-py")

# Import necessary libraries and packages
from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration

metric = evaluate.load("wer")

#Classes
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # First treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        #Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        #Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        #Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        #If bos token is appended in previous tokenization step,
        #Cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# Function
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    #Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    #We do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



#authenticate to huggingFace account

#load the Data
dar_voice = DatasetDict()

train_Data_Path = "./8dretna_daridja/data/train-00000-of-00001.parquet"
test_Data_Path  = "./8dretna_daridja/data/validation-00000-of-00001.parquet"

dar_voice["train"] = load_dataset("parquet", data_files=train_Data_Path)
dar_voice["test"] = load_dataset("parquet", data_files=test_Data_Path)

 # Remove unecessary attributes
dar_voice = dar_voice.remove_columns(['start_time', 'end_time'])


# Load the Model and prepare Feature Extractor and Tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="arabic", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="arabic", task="transcribe")

#Rescale the audio to 16KHZ
dar_voice = dar_voice.cast_column("audio", Audio(sampling_rate=16000))
#excute the rescaling function for all training data instances
dar_voice = dar_voice.map(prepare_dataset, remove_columns=dar_voice.column_names["train"], num_proc=2)

 #Initialise the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments

#finetuning parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="Models/whisperDAR",
    per_device_train_batch_size=8,      # Reasonable batch size for efficiency
    gradient_accumulation_steps=4,      # Effective utilization of GPUs
    learning_rate=3e-5,                  # Common learning rate for fine-tuning
    warmup_steps=300,                   # Conservative warmup steps
    max_steps=5000,                      # Utilize the 10-hour training time
    gradient_checkpointing=True,        # Save memory, recommended
    fp16=True,                          # Leveraging mixed precision training
    evaluation_strategy="steps",        # Evaluate every few steps
    per_device_eval_batch_size=8,       # Consistent with train batch size
    predict_with_generate=True,         # Enable generation during evaluation
    generation_max_length=50,           # Suitable length for generation
    save_steps=500,                     # Save model periodically
    eval_steps=500,                     # Evaluate every save step
    logging_steps=100,                  # Log metrics regularly
    report_to=["tensorboard"],
    load_best_model_at_end=True,        # Load the best model at the end
    metric_for_best_model="wer",        # Use WER for model selection
    greater_is_better=False,            # Lower WER is better                # Push model to Hub after training
    save_total_limit=5,                 # Limit the number of saved checkpoints
    num_train_epochs=25,                 # Adjusted for the total training time
)
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dar_voice["train"],
    eval_dataset=dar_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)

#start training
trainer.train()

kwargs = {
    "dataset_tags": "team4/8dretna_daridja",
    "dataset": "8dretna_daridja",  # a 'pretty' name for the training dataset
    "dataset_args": "split: test",
    "language": "ar",
    "model_name": "whisperDAR",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}



