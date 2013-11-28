# kaggle_seeclickfix

My model for the [See Click Predict Fix][1] Kaggle competition.  Read the
[postmortem][2] for a more detailed explanation of how it works.

## Generating a submission

1. Install dependencies `pip install -r requirements.txt`
2. Put data in _data/_ directory
3. Run model `python estimator.py`
4. Upload _submission.txt_

## Outcome

* 147/533 on the public leaderboard
* 151/533 on the private leaderboard

Puts me in the top 70%, which is not that great, but on the bright side, my
public and private leaderboard scores being very close means I didn't horribly
overfit during the competition.

[1]: http://www.kaggle.com/c/see-click-predict-fix
[2]: http://zacstewart.com/2013/11/27/kaggle-see-click-predict-fix-postmortem.html
