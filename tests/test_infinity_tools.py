import requests

import infinity_tools.sensefit as sensefit
import infinity_tools.visionfit as visionfit
import infinity_tools.sensefit.api as sa
import infinity_tools.visionfit.api as va


TOKEN = "TOKEN"


def test_version():
    def get_job_version(server: str, job_name: str) -> str:
        r = requests.get(
            f"{server}/api/jobs/",
            headers={"Authorization": f"Token {TOKEN}"},
        )
        return next((job["version"] for job in r.json() if job["name"] == job_name), None)

    assert sensefit.__version__ == get_job_version(sa.SERVER_URL, "sensefit")
    assert visionfit.__version__ == get_job_version(va.SERVER_URL, "visionfit")
