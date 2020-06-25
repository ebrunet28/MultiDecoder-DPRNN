# Questions:
1. If we use blackbox solver, do we pre-train the base extractor model, then train the blackbox model? Or is there a way that trains the base model along with the blackbox model?
2. What dataset do we use? The dataloader need to conform to the format. Do we do de-noising or source separation? Number of channels for these are different. What about batch size, sequence length? Do we use different audio combinations for each batch each epoch?
3. What method do we use to find the convergent solution?