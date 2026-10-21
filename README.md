# Hand Sign Lab

Teach your laptop your own hand signs with two classic models: logistic regression and a decision tree.
Made for the ReDI School machine learning course (classification).

**Open the lab:** https://alexonov.github.io/redi-hand-sign-lab/

- Runs fully in the browser. The camera image never leaves your laptop.
- A free Google model (MediaPipe) finds 21 points on your hand in each video frame. The page turns them into a small table of numbers. The models only ever see that table.
- Record signs → train → test on new recordings → choose a cutoff → play rock, paper, scissors.

**Python version:** download your data as CSV from the page, then open `hand_signs_starter.ipynb` in Colab or Databricks (`hand_signs_starter.py` is the same notebook as a plain script).
