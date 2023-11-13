# Assignment - 3: Q3

## Code Files Included
- 3_ad.py
- Inference_1.py
- Progression.py

## Text Files Included
- out.txt
- predictions.txt
- progression.txt

## Model Files Included
- decoder.pth
- encoder.pth
- decoder_0.pth
- encoder_0.pth
- decoder_0.5.pth
- encoder._0.5pth

## Directions For Operation
- This code was trained on ADA, with teacher forcing ratios to be 0, 1, and 0.5 respectively.(on 1080Ti's)
- The Best Results were obtained when the Teacher Forcing score was set to be 0.5.
- out.txt is the Train Log
- predictions.txt is the Validation Set outputs. 
- progression.txt is the outputs for progression.json, provided. 
- Inference_1, is used for inference and for beam search -> Only works with CUDA though.
- Same with Progression.json

### Everything is working quite well. 