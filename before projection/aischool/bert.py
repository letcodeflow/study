import sentencepiece as spm

parameter = '--input={} --model_prefix={} --vocab_size={} --model_type={}'

input_file = 'corpus.txt'
vocab_size = 32000
prefix = 'bert_kor'
model_type = 'bpe'
cmd = parameter.format(input_file, prefix, vocab_size, model_type)

# spm.SentencePieceTrainer.Train(cmd)

import sentencepiece as spm

parameter = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols={}'

input_file = 'corpus.txt'
vocab_size = 32000
prefix = 'bert_kor'
user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK]'
cmd = parameter.format(input_file, prefix, vocab_size,user_defined_symbols)

spm.SentencePieceTrainer.Train(cmd)

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('{}.model'.format(prefix))
token = sp.EncodeAsPieces('나는 오늘 아침밥을 먹었다.')
print(token)

# ['▁나는', '▁오늘', '▁아침', '밥을', '▁먹었다', '.']