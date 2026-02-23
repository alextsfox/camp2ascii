rm -rf .venv dist build camp2ascii.egg-info
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip build
python -m build
project_root="$(pwd)"
echo "Project root: $project_root"
pushd "$(mktemp -d)" && pip install "$project_root"/dist/*.whl
python3 -m unittest discover -s "$project_root/tests/" -p "test.py" -v
popd
deactivate
rm -rf .venv dist build camp2ascii.egg-info