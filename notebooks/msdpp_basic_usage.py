import torch

from msdpp.base.divmethod import DivDir
from msdpp.div_method.dpp import MSDPP

torch.manual_seed(42)  # For reproducibility

n_imgs = 100  # Number of images
n_attributes = 3  # Number of attributes
att_feat_dims = [10, 20, 30]  # Example dimensions for each attribute


# random scores between 0 and 1
scores = torch.rand((1, n_imgs))

# Image features of shape [n_imgs, n_feat_dim]
img_feats = torch.rand(n_imgs, 128)

# Attributes features of shape [n_imgs, n_att_dim]
atts_feats = [torch.rand(n_imgs, att_feat_dims[i]) for i in range(n_attributes)]


# tradeoff parameter for diversity vs. relevance
# Larger values lead to more focus on retrieval scores.
theta = 0.9

# tradeoff parameter for image feat vs attribute feats
# Larger values lead to more focus on image features.
beta = 0.5

# Whether to increase or decrease the diversity of atts_feats
direction = DivDir.DECREASE

msdpp = MSDPP(
    theta=theta,
    beta=beta,
)

reranked_index = msdpp.search(
    img_feats=img_feats,
    info=atts_feats,
    t2i_sim=scores,
    direction=direction,
)

# Print the reranked indices
print("The 10 highest ranked images are: ", reranked_index[0, :10].tolist())
