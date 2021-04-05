# BERT

## Masked Language Modeling

That's [mask] she [mask] -> That's what she said

## Next Sentence Prediction
**Input** = [CLS] That's [mask] she [mask]. [SEP] Hahaha, nice! [SEP]

**Label** = IsNext
* * *
**Input** = [CLS] That's [mask] she [mask]. [SEP] Dwight, you ignorant [mask]! [SEP]

**Label** = NotNext