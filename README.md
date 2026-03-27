# Integration 2 — PyTorch Housing Price Prediction

## What the model predicts
The model predicts **price_jod** based on these 5 features:
- area_sqm
- bedrooms
- floor
- age_years
- distance_to_center_km

## Training setup
- Model: Linear(5 → 32) → ReLU → Linear(32 → 1)
- Loss: MSELoss
- Optimizer: Adam (lr=0.01)
- Epochs: 100

## Training results
Loss decreased over time.
- Epoch 0: 1950591488.0000  
- Epoch 50: 1949066752.0000  
- Epoch 100: 1943020544.0000  

## Notes / observation
Loss drops gradually across epochs. Standardizing features helps training stability.