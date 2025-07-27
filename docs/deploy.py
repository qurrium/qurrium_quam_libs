"""This script is used to deploy the documentation to GitHub Pages using the ghp-import package."""

import os
import sys
import datetime
from ghp_import import ghp_import


with open("CNAME", "r", encoding="utf-8") as f:
    CNAME = f.read().strip()

CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"| Deploying documentation to GitHub Pages at {CURRENT_DATE}...")
print(f"| Custom domain: {CNAME}")


def deploy():
    """Deploy the documentation to GitHub Pages."""
    if not os.path.exists("./deploy"):
        print(
            "| Error: The deploy directory does not exist. "
            "Please create it or run the build command first."
        )
        sys.exit(1)

    ghp_import(
        "_build",
        nojekyll=True,
        force=True,
        push=True,
        cname=CNAME,
        mesg=f"ci: deploy to GitHub Pages - {CURRENT_DATE}",
        branch="gh-pages",
    )
    print("| Deployment complete.")


if __name__ == "__main__":
    deploy()

    print("| Deployment complete.")
