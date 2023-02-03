# %%
import re

s = "green greggs gand gam"
# Python regexes work as follows: We have a pattern and a replacement. Replacement is either a string or a function.
# Index at 1, group 0 is the entire match
print(re.sub(r"(\w) \w", " \g<1>\g<0>", s))
s = "greeN GreggS gand gam"
print(
    re.sub(
        r"(\w) (\w)",
        lambda match: f"{match.group(2).lower()} {match.group(1).lower()}",
        s,
    )
)


# %%
