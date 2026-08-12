import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from config import get_weights_file_path, get_config
from torch.utils.tensorboard import SummaryWriter
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from pathlib import Path
import os
import warnings



# Function to extract sentences for training the tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]



# Function to generate translations using greedy decoding

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer




# Function to generate translations using greedy decoding

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)
        
        if next_word.item() == eos_idx:
            break
    
    return decoder_input.squeeze(0)



# Function to validate the model after each epoch

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            print_msg('-'*console_width)
            print_msg(f"SOURCE: {batch['src_text'][0]}")
            print_msg(f"TARGET: {batch['tgt_text'][0]}")
            print_msg(f"PREDICTED: {model_out_text}")
            
            if count == num_examples:
                print_msg('-'*console_width)
                break





# Function to prepare dataset and tokenizers

def get_ds(config):
    ds_raw = load_dataset("Helsinki-NLP/opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train") ### Put your dataset here
    ds_list = list(ds_raw)
    
    tokenizer_src = get_or_build_tokenizer(config, ds_list, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_list, config["lang_tgt"])
    
    train_size = int(0.9 * len(ds_list))
    val_size = len(ds_list) - train_size
    train_ds, val_ds = random_split(ds_list, [train_size, val_size])
    
    train_dataloader = DataLoader(BilingualDataset(train_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len']), batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(BilingualDataset(val_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len']), batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


# Function to initialize the Transformer model

def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])


# Main training loop

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch, global_step = 0, 0
    
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        if os.path.exists(model_filename):
            print(f"Preloading model {model_filename}")
            state = torch.load(model_filename)
            initial_epoch = state['epochs'] + 1
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        else:
            print(f"Warning: Preload file {model_filename} not found. Training from scratch.")
    
    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            
            label = batch['label'].to(device)
            loss = loss_function(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        torch.save({'epochs': epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'global_step': global_step}, get_weights_file_path(config, f"{epoch:02d}"))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    
