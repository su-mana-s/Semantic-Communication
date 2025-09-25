## The What
Semantic Message Extraction for Text Based Data With Deep Neural Nets
- The project aims to semantically encode text and extract meaning to aid in transmission as proposed in [1] and [2] and is built with pyTorch.
----------
## The Why
- Traditional communication systems have focused on improving the transmission rates through physical channels. They are constrained by the Shannon Limit which defines the maximum data rate that can be achieved over a channel. While this theoretical limit has not been reached, the current systems are close and 6g communication research proposes alternate transmission methods.
- Semantic communication is one such model which extracts the semantic message embedding from the original message, thus only necessitating the transfer of essential information and discarding the trivial. It follows logically then, that  this enables the transmission of more information. Current research in the field employs Deep Learning architectures to achieve this extraction of meaning. While this is relevant to different modalities of data, the focus of this project is text.  
----------
## The How
- Multiple architectures have been proposed in literature, from LSTMs to Transformers[1] and AutoEncoders[2]. The aim of the system is to obtain c = f(m) where "c" is the semantic message extracted from the original "m" and "f" is a deep learning model that facilitates this conversion. BLEU scores (sentence BLEU) are used as the performance metric. 

### The Data:
[Europarl: A Parallel Corpus for Statistical Machine Translation](https://www.statmt.org/europarl/) offers sentence-aligned text for machine translation. Single-language data is sufficient for our purposes, since our aim here is to merely recreate the original message on the receiver's end. To this end, the English section of the "bg-en" dataset was used.

### The Project:
1) Data Pre-Processing & Dataset creation 
- Data: EuroParl corpus(bg-en)
- Samples: Training - 233093, Testing - 25899, Batch size 128
2) Model Training
- Indexing, Embedding
- Semantic Encoding, Semantic Decoding (Transformers), greedy decoding
-  Seq2Txt - Tensor to Text from Vocab
- Loss - CE Loss - Cross Entropy
3) Performance metric - BLEU score
- Sentence BLEU between encoder-decoder output and original text with 1 gram, 2 gram, 3 gram, 4 gram and an average with equal weights(*0.25) scores were calculated.

### The Model:
![arch](https://github.com/user-attachments/assets/732d5929-aec4-45ff-a0f3-e6e44591eca0)


<!-- ## Quick Start -->
<!-- 1) Clone the project
```
git clone https://github.com/su-mana-s/Ramayanam.git
```
2) Open command line and navigate to the project folder streamlit source folder - Ramayanam/src/
```
cd Ramayanam/src
```
3) Run the app
```
streamlit run Jignyasa.py
``` -->
----------
## Outputs
- Training Epochs - 12
- Testing Epochs - 2
- Test cases - 25899
1) BLEU scores <br><br>
![BLEU-LR scores](https://github.com/user-attachments/assets/b7ce562c-c70b-47b1-af43-5c9ae435b7ac)

3) Sentences  - After Semantic Extraction & Decoding <br><br>
![image](https://github.com/user-attachments/assets/352f1126-20c5-4fbc-833e-d35e8dc57a80)

----------
## References
[1] Huiqiang Xie et.al., ”Deep Learning Enabled Semantic Communication Systems,”IEEE Transactions on signal processing, vol 69, 2021.

[2] Xinlai Luo, “Autoencoder-based Semantic Communication Systems with Relay Channels,” arXiv:2111.10083v1 [cs.IT] 19 Nov 2021
