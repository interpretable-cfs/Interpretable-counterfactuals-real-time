This is the repository with the source code of the paper: Interpretable and Efficient Counterfactual Generation for real-time User Interaction. 

To replicate the results of our study you need to follow these steps:
1) Train a disentangled RAE with the code provided in architectures-py, models.py and training.py
2) Extract concepts via latent traversal of the class medoids. Code in medoids.py and latent_traversal.py allow you to do so. You should store a dictionary of concepts as a .josn file in your directory. An example of this dictionary is provided at the beginning of explanations.py. Do not use that dictionary for your explanations as your model will certainly assign those concepts to different latent dimensions.
3) Generate explanations with explanations.py. The outputs ia a plot with the original image and the label in the first column. The remaining columns correspond to counterfactuals with associated concepts for each of the asked classes.

