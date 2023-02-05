import datetime
import time
from datetime import timezone

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

COMPETITION = "rsna-breast-cancer-detection"
result_ = api.competition_submissions(COMPETITION)[0]
latest_ref = str(result_)  # 最新のサブミット番号
submit_time = result_.date
print("サブミット番号", latest_ref)
print("サブミット時間", submit_time + datetime.timedelta(hours=9))

status = ""

while status != "complete":
    list_of_submission = api.competition_submissions(COMPETITION)
    for result in list_of_submission:
        if str(result.ref) == latest_ref:
            break
    status = result.status

    now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed_time = int((now - submit_time).seconds / 60) + 1
    if status == "complete":
        print("\r", f"run-time: {elapsed_time} min, LB: {result.publicScore}")
    else:
        print("\r", f"elapsed time: {elapsed_time} min", end="")
        time.sleep(60)
