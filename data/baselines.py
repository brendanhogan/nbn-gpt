"""
Holds baseline accuracy/loss values - used in training code for logging, and plotting code. 

"""

loss_baseline = {"124M": 3.2924}
hella2_baseline = { # HellaSwag for GPT-2
    "124M": 0.294463,
    "1558M": 0.488946,
}
hella3_baseline = { # HellaSwag for GPT-3
    "124M": 0.337,
    "1558M": 0.547,
}