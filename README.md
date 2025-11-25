# **A Geometric-Constraint Driven Latent Space Prototype Learning Framework for One-Shot Classification** **(GCLP)**

## 1. Files Introduction

1. The "Checkpoint" is used to save the trained model. The model we proposed is saved in the "gclp" folder. To facilitate code reproduction, we provide four trained models: 3-1-1e-5.pth.tar, 5-1-1e-5.pth.tar, 7-1-1e-5.pth.tar, and 9-1-1e-5.pth.tar. You can download the model at https://pan.baidu.com/s/1MO19crjY2QezFrWcKIHdFA?pwd=ztku.
2. The datasets we use is in the "data". It includes the training set "train_fewrel", the validation set "val_fewrel", and three test sets "val_fewrel" (named Val in the paper), "test_fewrel"  (named FewRel1.0 in the paper), and "val_pubmed_new" (named FewRel2.0 in the paper). There are two files containing the class names and descriptions. The "pid2name" corresponds to fewrel1.0 dataset. The "newpidname" corresponds to fewrel2.0 dataset.
3. In the "fewshot_re_kit" are related configuration files, including "data_loader", "sentence_encoder", etc.
4. All models are saved in the "models". The model we proposed is marked as "gclp".
5. The "Pretrain" saves configuration files related to embedding and encoding. Download from https://pan.baidu.com/s/1VThkF9ieBEEeuLZWKJW0Xw?pwd=qqi9.

## 2. Train

You can train the model using the following command.

```
sh run_train.sh
```

Alternatively, you can also train the model using the following python command.

```
python train_demo.py \
    --trainN 3 --N 3 --K 1 --Q 1 --dot \
    --model gclp --encoder bert --hidden_size 768 --val_step 2000 --lr 1e-5 \
    --pretrain_ckpt pretrain \
    --batch_size 2 --save_ckpt checkpoint/gclp/3-1-1e-5.pth.tar \
    --cat_entity_rep \
    --backend_model bert
```

However, we recommend the first command. Note that when you want to modify the external information used, you need to make the following changes in file "sentence_encoder".

```
class BERTSentenceEncoder(nn.Module):
...
	def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        # [CLS] + label words + [SEP]
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')

        # [CLS] label words + [SEP] + label descriptions
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
            ...
```

## 3. Test

We used three test sets. You can test the model using the following command.

```
sh run_eval.sh
```

Alternatively, you can also test the model using the following python command.

```
python test_demo.py \
    --trainN 3 --N 3 --K 1 --Q 1 --dot \
    --model gclp --encoder bert --hidden_size 768 --val_step 2000 --test test_fewrel \
    --batch_size 2 --only_test \
    --load_ckpt checkpoint/checkpoint/gclp/3-1-1e-5.pth.tar \
    --pretrain_ckpt pretrain \
    --cat_entity_rep \
    --test_iter 1000 \
    --backend_model bert
```

However, we recommend the first command. 

1. If you want to use the first test set to test the model's data fitting ability, you need to set the value of "--test" to "val_fewrel".

2. If you want to use the second test set to test the model's intra-domain generalization ability, you need to set the value of "--test" to "test_fewrel".

3. If you want to use the thrid test set to test the model's cross-domain generalization ability, you need to set the value of "--test" to "val_pubmed_new". Note that when you use dataset "val_pubmed_new" for testing, you need to make the following changes in file "data_loader".

   ```
   # use fewrel1.0
   # pid2name = 'pid2name'
   # pid2name_path = os.path.join(root, pid2name + ".json")
   
   # use fewrel2.0 for test
   pid2name = 'newpidname'
   pid2name_path = os.path.join(root, pid2name + ".json")
   ```

## 4. Results

|    Dataset    | 3-way-1-shot | 5-way-1-shot | 7-way-1-shot  | 9-way-1-shot  |
| :------------ | :----------: | :----------: | :-----------: | :-----------: |
|     Val       |     97.83    |     95.88    |     93.81     |     92.88     |
| FewRel1.0     |  94.31±0.31  |  91.05±0.24  |  87.89±0.16   |  85.91±0.20   |
| FewRel2.0     |  82.93±0.22  |  77.75±0.23  |  75.02±0.34   |  69.92±0.06   |



## 5. Cite

If you would like to cite this article or use data published in this article, please use the citation format uploaded later.
