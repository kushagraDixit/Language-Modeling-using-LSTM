from scripts import utils
import pickle
import torch.nn as nn
from tqdm import tqdm
import torch.nn as nn
import torch
import math
import numpy as np
import sys
import argparse

class CreateDataset():
    def __init__(self,vocab_path,train_path,dev_path,test_path,seq_length,batch_size,device) -> None:
        self.vocab = self.get_vocab(vocab_path)
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.seq_length = seq_length
        self.batch_size = batch_size

        self.train_tokens = self.get_tokanized_data(train_path,self.vocab)
        self.dev_tokens = self.get_tokanized_data(dev_path,self.vocab)
        self.test_tokens = self.get_tokanized_data(test_path,self.vocab)

        self.weight_tensor = self.get_weight_vector(self.train_tokens,self.vocab,device)

        print("Creating Training Input and Output Dataset : ")
        self.training_data_input,self.training_data_output = self.get_data(self.train_tokens,self.seq_length,self.vocab)
        print("Creating Dev Input and Output Dataset : ")
        self.dev_data_input,self.dev_data_output = self.get_data(self.dev_tokens,self.seq_length,self.vocab)
        print("Creating Test Input and Output Dataset : ")
        self.test_data_input,self.test_data_output = self.get_data(self.test_tokens,self.seq_length,self.vocab)

        print("Creating All DataLoaders : ")
        self.train_loader = self.get_dataloader(self.training_data_input,self.training_data_output,self.batch_size,device)
        self.dev_loader = self.get_dataloader(self.dev_data_input,self.dev_data_output,self.batch_size,device)
        self.test_loader = self.get_dataloader(self.test_data_input,self.test_data_output,self.batch_size,device)

        pass

    def get_vocab(self,vocab_path):
        with open (vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        return vocab
    
    def get_tokanized_data(self,data_path,vocab):
        data_files = utils.get_files(data_path)
        data_tokens = utils.convert_files2idx(data_files,vocab)
        return data_tokens
    
    def get_weight_vector(self,train_tokens,vocab,device):
        print("Getting Weight vector : ")
        weights = np.zeros(len(vocab),dtype=float)
        for i in tqdm(range(len(train_tokens))):
            for j in range(len(train_tokens[i])):
                weights[train_tokens[i][j]] = weights[train_tokens[i][j]] + 1

        weights = 1 - (weights/np.sum(weights))
        weights = weights.astype(np.float32)
        weights = torch.from_numpy(weights).to(device)

        return weights
    
    def get_data(self,data_tokens,seq_length,vocab):
        input_data = []

        for i in tqdm(range(len(data_tokens))):
            
            length = len(data_tokens[i])
            
            while length/seq_length>1:
                row = np.asarray(data_tokens[i][:seq_length])
                input_data.append(row)
                data_tokens[i] = data_tokens[i][seq_length:length]
                length = len(data_tokens[i])
                
            if length > 0:
                rem_len = seq_length - length
                row = np.asarray(data_tokens[i])
                rem_arr = np.full(rem_len,vocab['[PAD]'])
                row = np.concatenate((row,rem_arr))
                input_data.append(row)

        output_data = []
        for i in tqdm(range(len(input_data)-1)):
            out = np.concatenate((input_data[i][1:len(input_data[i])],np.asarray([input_data[i+1][0]])))
            output_data.append(out)

        out = np.concatenate((input_data[len(input_data)-1][1:len(input_data[len(input_data)-1])],np.asarray([vocab['[PAD]']])))
        output_data.append(out)
        
        return np.array(input_data),np.array(output_data)
    
    def get_dataloader(self,input_array,output_array,batch_size,device):
        input_data = torch.tensor(input_array, dtype=torch.long).to(device)
        output_data = torch.tensor(output_array, dtype=torch.long).to(device)

        dataset = torch.utils.data.TensorDataset(input_data, output_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,drop_last=True)

        return dataloader

class N_Gram_Model():
    def __init__(self,train_tokens, test_tokens, n_grams) -> None:
        self.prob_dist = {}
        self.prob_dist_history = {}
        self.n_grams = n_grams
        
        print("Creating Dataset for N-Grams Model : ")
        print("Processing Training Data : ")
        self.training_data_n_grams = self.get_processed_data(train_tokens)
        print("Processing Test Data : ")
        self.test_data_n_grams = self.get_processed_data(test_tokens)
        pass     
    
    def get_processed_data(self,tokenized_data):
        n_gram_data = []
        for i in tqdm(range(len(tokenized_data))):
            list_elem = [384,384,384]
            list_elem.extend(tokenized_data[i])
            n_gram_data.append(list_elem)

        return n_gram_data
    
    def train(self):
        print("\nTraining N-Gram Model..... ")
        for i in tqdm(range(len(self.training_data_n_grams))):
            #print(len(training_data_n_gram)-n_gram+1)
            for j in range(len(self.training_data_n_grams[i])-self.n_grams+1):

                key_history = ""
                
                key_history = str(self.training_data_n_grams[i][j])+str(self.training_data_n_grams[i][j+1])+str(self.training_data_n_grams[i][j+2])
                pred_char = self.training_data_n_grams[i][j+3]
                

                key = key_history+":"+str(pred_char)

                if key_history in self.prob_dist_history:
                    self.prob_dist_history[key_history] = self.prob_dist_history[key_history] + 1
                else:
                    self.prob_dist_history[key_history] = 1
        
                if key in self.prob_dist:
                    self.prob_dist[key] = self.prob_dist[key] + 1
                else:
                    self.prob_dist[key] = 1

    def find_test_perplexity(self,vocab):
        print("Calculating Test Perplexity for N-Gram Model..... ")
        perplexity = []
        for i in tqdm(range(len(self.test_data_n_grams))):
            loss = []
            for j in range(len(self.test_data_n_grams[i]) - self.n_grams + 1):
                
                key_history = ""

                key_history = str(self.test_data_n_grams[i][j])+str(self.test_data_n_grams[i][j+1])+str(self.test_data_n_grams[i][j+2])
                pred_char = self.test_data_n_grams[i][j+3]
            

                key = key_history+":"+str(pred_char)
                
                if key in self.prob_dist:
                    prob_dist_pred_char = (self.prob_dist[key] + 1)/(self.prob_dist_history[key_history] + len(vocab)) 
                else:
                    if key_history in self.prob_dist_history:
                        prob_dist_pred_char = 1/(self.prob_dist_history[key_history] + len(vocab))
                    else:
                        prob_dist_pred_char = 1/len(vocab)
                
            
                loss.append(math.log2(prob_dist_pred_char))
            
            loss = -1*np.mean(loss)

            perp = math.pow(2,loss)

            perplexity.append(perp)

        print(f" Test Perplexity of {self.n_grams} Model is : {np.mean(perplexity)}")


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,seq_length):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.reLU = torch.nn.ReLU()
        self.W2 = nn.Linear(hidden_dim, vocab_size, bias=True)

    def forward(self, input, hidden,target=None):

        embedding = self.embedding(input)
        output, hidden = self.lstm(embedding, hidden)
        prediction = self.reLU(self.W1(output))
        prediction = self.W2(prediction)
        return prediction, hidden
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        #print(f"Hidden: {hidden.size()}")
        return hidden, cell

def train_language_model(model,optimizer,loss_fn_train,loss_fn_evaluate,train_loader,dev_loader,seq_length,batch_size,max_epochs,learning_rate,seed):
    print("\nTraining Language Model..... ")
    best_perf_dict = {"perplexity": sys.maxsize, "epoch": 0, "dev_loss": sys.maxsize}

    for ep in range(1, max_epochs + 1):
        print(f"Epoch {ep}")
        train_loss = []
        h = model.init_hidden(batch_size, device)
        
        # Training Loop

        for inp, lab in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            h = model.detach_hidden(h)
            out, h = model(inp.to(device),h,lab.to(device))
            out = out.reshape(batch_size*seq_length,-1)
            lab = lab.reshape(-1)
            loss = loss_fn_train(out, lab.to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())

        print(f"Average training batch loss: {np.mean(train_loss)}")

        dev_loss = []
        perplexities = []
        dev_loss = []
        
        for inp, lab in tqdm(dev_loader):
            model.eval()
            with torch.no_grad():
                h = model.detach_hidden(h)
                out, h = model(inp.to(device),h,lab.to(device))
                out = out.reshape(batch_size*seq_length,-1)
                lab = lab.reshape(-1)
                loss = loss_fn_evaluate(out, lab.to(device))
                loss = torch.reshape(loss,shape=(batch_size,seq_length))
                batch_loss = []
                for seq in loss:
                     seq_loss = torch.mean(seq).item()
                     perplexity = math.pow(2,seq_loss)
                     perplexities.append(perplexity)
                     batch_loss.append(seq_loss)

                dev_loss.append(np.mean(batch_loss))

        dev_loss_value = np.mean(dev_loss)
        print(f"Average dev batch loss: {dev_loss_value}")

        perplexity_value = np.mean(perplexities)
        print(f"Average dev perplexity: {perplexity_value}")

        if perplexity_value<best_perf_dict['perplexity']:
            best_perf_dict['perplexity'] = perplexity_value
            best_perf_dict['epoch'] = ep
            best_perf_dict['dev_loss'] = dev_loss_value

            torch.save({
                "model_param": model.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_perplexity": best_perf_dict['perplexity'],
                "dev_loss": dev_loss_value,
                "epoch": ep
            }, f"./Models/{seed}models_bs{batch_size}_lr{learning_rate}_ep{ep}")

            best_path = f"./Models/{seed}models_bs{batch_size}_lr{learning_rate}_ep{ep}"
    
    print(f"""\nBest perplexity of {best_perf_dict['perplexity']} at epoch {best_perf_dict["epoch"]}""")
    print(f"""\nDev Loss of {best_perf_dict['dev_loss']} at epoch {best_perf_dict["epoch"]}""")

    return best_path

def get_string_from_indices(indices,inv_vocab):
    res = ""
    for i in indices:
        res = res + str(inv_vocab[i])

    return res

def generate_text(sentence, gen_seq_len, temp, model, vocab, inv_vocab, device, seed):
    torch.manual_seed(seed)
    model.eval()
    indices = [vocab[t] for t in sentence]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(gen_seq_len):
            src = torch.LongTensor([indices]).to(device)
            pred, hidden = model(src, hidden)
            probabilities = torch.softmax(pred[:, -1] / temp, dim=-1)  
            pred = torch.multinomial(probabilities, num_samples=1).item()    
            
            while pred == vocab['<unk>'] or pred == vocab['[PAD]']:
                pred = torch.multinomial(probabilities, num_samples=1).item()

            indices.append(pred)

    tokens = get_string_from_indices(indices,inv_vocab)
    return tokens

def get_test_perplexity_and_generate_text(model,model_path, optimizer, loss_fn_evaluate,test_loader, prompts, batch_size, seq_length,seed):
    print("\nCalculating Test Perplexity for Language Model..... ")
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint["model_param"])
    optimizer.load_state_dict(checkpoint["optim_param"])
    
    test_loss = []
    perplexities = []
    test_loss = []

    h = model.init_hidden(batch_size, device)
        
    for inp, lab in tqdm(test_loader):
        model.eval()
        with torch.no_grad():
            h = model.detach_hidden(h)
            out, h = model(inp.to(device),h,lab.to(device))
            out = out.reshape(batch_size*seq_length,-1)
            lab = lab.reshape(-1)
            loss = loss_fn_evaluate(out, lab.to(device))
            loss = torch.reshape(loss,shape=(batch_size,seq_length))
            batch_loss = []
            for seq in loss:
                seq_loss = torch.mean(seq).item()
                perplexity = math.pow(2,seq_loss)
                perplexities.append(perplexity)
                batch_loss.append(seq_loss)

            test_loss.append(np.mean(batch_loss))

    test_loss_value = np.mean(test_loss)
    print(f"Average Test batch loss: {test_loss_value}")

    perplexity_value = np.mean(perplexities)
    print(f"Average Test perplexity: {perplexity_value}")

    print("\nGenerating Text for Prompts..... ")
    temp = 0.5
    for i, sentence in enumerate(prompts):
        res_text = generate_text(sentence,200, temp, model,dataset.vocab, dataset.inv_vocab, device,seed)

        print(f"Text{i+1} : {res_text} \n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.001, type=float, help='The Learning Rate of the Model')
    parser.add_argument('--batch_size', default=64, type=int, help='The batch size for training')
    parser.add_argument('--hidden_dim', default=200, type=int, help='Dimension of hidden vector of LSTM')
    parser.add_argument('--embedding_dim', default=50, type=int, help='Dimensions for embedding layer')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of layers of LSTM')
    parser.add_argument('--seq_length', default=500, type=int, help='Length of input sequence to LSTM')
    parser.add_argument('--vocab_path', default='./data/vocab.pkl', type=str, help='Path to vacabulary')
    parser.add_argument('--train_path', default='./data/train', type=str, help='Directory where the training data is stored')
    parser.add_argument('--dev_path', default='./data/dev', type=str, help='Directory where the dev data is stored')
    parser.add_argument('--test_path', default='./data/test', type=str, help='Directory where the test data is stored')
    parser.add_argument('--epochs', default=5, type=int, help='Total number of epochs for training')
    parser.add_argument('--n_grams', default=4, type=int, help='N-grams for N-GRAM Model')
    parser.add_argument('--seed', default=24, type=int, help='Initial seed')

    args = parser.parse_args()
    LEARNING_RATE , MAX_EPOCHS, BATCH_SIZE , HIDDEN_DIM, EMBEDDING_DIM , NUM_LAYERS, SEQ_LEN, VOCAB_PATH, TRAIN_PATH , DEV_PATH, TEST_PATH, N_GRAMS, SEED = (args.learning_rate , args.epochs,
                                                                                                                args.batch_size , args.hidden_dim, args.embedding_dim, 
                                                                                                                args.num_layers, args.seq_length, args.vocab_path,
                                                                                                                args.train_path, args.dev_path, args.test_path, args.n_grams, args.seed)

    torch.manual_seed(SEED)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    dataset = CreateDataset(VOCAB_PATH,TRAIN_PATH,DEV_PATH,TEST_PATH,SEQ_LEN,BATCH_SIZE,device)

    prob_model = N_Gram_Model(dataset.train_tokens,dataset.test_tokens,N_GRAMS)

    prob_model.train()

    prob_model.find_test_perplexity(dataset.vocab)

    print(f"\nNumber of Parameters (number of conditional probabilities) for N-gram Model : {len(prob_model.prob_dist)}")
    
    model = LanguageModel(dataset.vocab_size,EMBEDDING_DIM,HIDDEN_DIM,NUM_LAYERS,SEQ_LEN).to(device)

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    loss_fn_train = torch.nn.CrossEntropyLoss(weight=dataset.weight_tensor,ignore_index=dataset.vocab['[PAD]'])

    loss_fn_evaluate = torch.nn.CrossEntropyLoss(weight=dataset.weight_tensor,ignore_index=dataset.vocab['[PAD]'],reduce=False)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nThe model has {num_params:,} trainable parameters')

    best_model = train_language_model(model,optimizer,loss_fn_train,loss_fn_evaluate,dataset.train_loader,dataset.dev_loader,SEQ_LEN,BATCH_SIZE,MAX_EPOCHS,LEARNING_RATE,SEED)
    
    prompts = [ "The little boy was" , "Once upon a time in" , "With the target in" , "Capitals are big cities. For example," , "A cheap alternative to" ]

    get_test_perplexity_and_generate_text(model,best_model,optimizer,loss_fn_evaluate,dataset.test_loader,prompts,BATCH_SIZE,SEQ_LEN,SEED)

