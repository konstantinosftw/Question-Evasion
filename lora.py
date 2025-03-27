import torch
import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset 
import transformers
import argparse
import pandas as pd 

class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


mapping ={
    '1.1 Explicit': 'Direct Reply',
    '1.2 Implicit': 'Indirect',
    '2.1 Dodging': "Indirect",
    '2.2 Deflection': "Indirect",
    '2.3 Partial/half-answer': "Indirect",
    '2.4 General': "Indirect",
    '2.6 Declining to answer': "Direct Non-Reply",
    '2.7 Claims ignorance': "Direct Non-Reply",
    '2.8 Clarification': "Direct Non-Reply",
}


def load_qevasion_dataset(tokenizer, train_size = 900, annotator_ids = None, add_specific_labels = False):
    
    dataset = pd.read_csv("dataset/QAEvasion.csv")

    texts = []
    for _, row in dataset.iterrows():

        if annotator_ids != None and row["annotator_id"] not in annotator_ids:
            continue 

        if row["label"] not in mapping:
            continue
            
        text = "Based on a part of the interview where the interviewer asks a set of questions, classify the type of answer the interviewee provided for the following question.\n\n ### Part of the interview ###\n" + row["interview_question"] + "\n" + row["interview_answer"] + "\n\n" + "### Question ###\n\n"
        
        if add_specific_labels:
            text += row["question"] + "\nLabel: " + row["label"] + "\n\n"
        else:
            text += row["question"] + "\nLabel: " + mapping[row["label"]] + "\n\n"
        texts.append(text)

    print (texts[1])
    return texts, CustomTextDataset(texts[:train_size], tokenizer)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(model_name, train_size, annotator_ids, output_model_dir, add_specific_labels):

    cache_dir = ""

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map='auto',
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True,)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(
        r=16, #attention heads
        lora_alpha=32, #alpha scaling
        #target_modules=["q_proj", "v_proj"], #if you know the
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # load data
    texts, data = load_qevasion_dataset(tokenizer, train_size = train_size, annotator_ids = annotator_ids, add_specific_labels=add_specific_labels)

    print (f"Found {len(data)} instances for training and {len(texts) - len(data)} instances for validation.")

    # train model
    print ("Training . . . ")
    out_dir = output_model_dir.split("/")[-1]
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            max_steps=len(data) * 5,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir=f'outputs_{out_dir}'
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    # Save the model
    model.save_pretrained(output_model_dir)

    # Optionally, save the tokenizer as well
    tokenizer.save_pretrained(output_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with arguments.")
    parser.add_argument("--model_name", type=str, help="Name of the model e.g. bigscience/bloom-3b")
    parser.add_argument("--train_size", type=int, help="Number of instances in  the training dataset", default = 900)
    parser.add_argument("--annotator_ids", nargs="*", type=int, default=None, help="Ids of the annotators that will be used during training. Default value None, which means all the instances will be used, regardless the annotator!")
    parser.add_argument("--output_model_dir", type=str, help="Directory to save the trained model")
    parser.add_argument("--add_specific_labels", action="store_true", help="Include this flag to indicate whether specific labels (e.g. General, Partial etc) should be added or not")


    args = parser.parse_args()

    print (args.model_name, args.train_size, args.annotator_ids, args.output_model_dir, args.add_specific_labels)
    
    print (type(args.add_specific_labels))
    main(args.model_name, args.train_size, args.annotator_ids, args.output_model_dir, args.add_specific_labels)
