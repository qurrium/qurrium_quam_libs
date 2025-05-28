# Tools

---

## `set_version_3div.py`

Get the version number from the VERSION.txt file and pass it to the environment variable.

- This script is shared with :package:`qurrium` and :package:`qurecipe` repositories.

### Release - `-r`, `--release`

- `stable`

Set version number to the stable version number like `0.x.x` which removed the `dev` part.
And add this version number to the git tag.

Also, when use `stable`, any bump (`-b`, `--bump`) movement except `skip` was not allowed.
It will raise a `ValueError` to terminate workflow.

- `check` - the default argument

The versioning checking is used for unit test before the pull request.
Confirm that the version number is still the last version in git version.
Do not modify by develop and ready for bumping version after pull request.

Also, when use `check`, any bump (`-b`, `--bump`) movement except `skip` will take no effect.
It will raise a warning to mention.

### Bump - `-b`. `--bump`

- `skip` - the default argument

Keep current version nummber.

- `patch`

Bump the patch version number. For example, `0.3.1` to `0.3.2`.

- `minor`

Bump the minor version number. For example, `0.3.1` to `0.4.0`.

- `major` - Not implemented

Bump the major version number. For example, `0.3.1` to `1.0.0`.

### Test - `-t`, `--test`

Test script and do nothing on versioning.
