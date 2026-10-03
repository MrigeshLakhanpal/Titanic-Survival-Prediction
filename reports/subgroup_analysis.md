# Subgroup Accuracy Analysis

Accuracy computed from out-of-fold predictions (cross_val_predict), so every passenger is scored by a model that never trained on that passenger.

## By Sex

 | Sex | Accuracy| n |
|---|---|---|
| female | 0.7962 | 314 | 
| male | 0.8215 | 577 | 

## By Passenger Class

 | Pclass | Accuracy| n |
|---|---|---|
| 1 | 0.7685 | 216 | 
| 2 | 0.9239 | 184 | 
| 3 | 0.7902 | 491 | 

## By Age Bracket

 | AgeBracket | Accuracy| n |
|---|---|---|
| Adult (>=16) | 0.8193 | 631 | 
| Child (<16) | 0.7349 | 83 | 
| Unknown | 0.8249 | 177 | 

**Overall accuracy (all rows, out-of-fold)**: 0.8126

Compare each subgroup's accuracy against this overall number -- a subgroup sitting noticeably below it is where the model is weakest, even if the aggregate score looks fine.