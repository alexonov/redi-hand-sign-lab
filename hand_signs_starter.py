# %% [markdown]
# # Hand Sign Lab — the same models in Python
#
# In the browser you recorded your hand signs and pressed **Download data as CSV**.
# Upload `hand_signs.csv` next to this notebook (Colab: folder icon on the left → upload;
# Databricks: upload to your workspace folder) and run the cells top to bottom.
#
# Each row is one camera frame. A Google model (MediaPipe) found 21 points on your hand.
# From them we computed 6 finger measurements and 42 raw point positions. That table is
# all the models ever see. No pictures.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv("hand_signs.csv")
print(data.groupby(["set", "label"]).size())
data.head()

# %% [markdown]
# ## 1. Features and target
# Start with the 6 finger measurements. Later, try the 42 raw positions instead.

# %%
FINGER_FEATURES = ["thumb out", "index stretch", "middle stretch", "ring stretch", "pinky stretch", "index–middle gap"]
RAW_FEATURES = [c for c in data.columns if c.endswith(" x") or c.endswith(" y")]

features = FINGER_FEATURES          # try: RAW_FEATURES
train = data[data["set"] == "train"]
test = data[data["set"] == "test"]  # recorded differently, never used for training

X_train, y_train = train[features], train["label"]
X_test, y_test = test[features], test["label"]

# %% [markdown]
# ## 2. Two models, two scores
# Training score = how well it remembers. Test score = how well it learned.

# %%
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0)

for name, model in [("logistic regression", logreg), ("decision tree", tree)]:
    model.fit(X_train, y_train)
    print(f"{name:20}  train {accuracy_score(y_train, model.predict(X_train)):.0%}"
          f"   test {accuracy_score(y_test, model.predict(X_test)):.0%}")

# %% [markdown]
# ## 3. Look inside
# The tree is a set of if-then rules. Logistic regression is one weight per feature per sign.

# %%
plt.figure(figsize=(14, 6))
plot_tree(tree, feature_names=features, class_names=list(tree.classes_), filled=True, rounded=True, fontsize=9)
plt.show()

weights = pd.DataFrame(logreg[-1].coef_.T, index=features, columns=logreg[-1].classes_)
weights.round(2)

# %% [markdown]
# ## 4. The cutoff: act only when sure
# `predict_proba` gives a probability per sign. Below the cutoff the model says "not sure".

# %%
CUTOFF = 0.8
model = logreg  # try: tree

proba = model.predict_proba(X_test)
best = proba.argmax(axis=1)
said = pd.Series(model.classes_[best], index=X_test.index)
said[proba.max(axis=1) < CUTOFF] = "not sure"

print(pd.crosstab(y_test, said, rownames=["showed"], colnames=["said"]))
acted = said != "not sure"
print(f"\nacts on {acted.mean():.0%} of test rows, right {(said[acted] == y_test[acted]).mean():.0%} of the time when it acts")

# %% [markdown]
# ## 5. Tree depth: learning vs memorising

# %%
rows = []
for depth in [1, 2, 3, 4, 6, 8, 12]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    rows.append({"max_depth": depth,
                 "train": accuracy_score(y_train, t.predict(X_train)),
                 "test": accuracy_score(y_test, t.predict(X_test))})
depth_scores = pd.DataFrame(rows).set_index("max_depth")
depth_scores.plot(marker="o", ylabel="accuracy", title="Tree depth: train vs test")
plt.show()
depth_scores.round(2)

# %% [markdown]
# ## Your turn
# 1. Switch `features` to `RAW_FEATURES`. Which model gets better, which gets worse? Why?
# 2. Invent a new feature from the raw points, e.g. distance between thumb tip and index tip
#    (`thumb tip x`, `thumb tip y`, `index tip x`, `index tip y`). Does the test score move?
# 3. Swap CSVs with a teammate: train on yours, use theirs as the test.
# 4. Try `RandomForestClassifier` from `sklearn.ensemble`. Next session is about why it is stronger.
