./update.sh

rm dist/*
python setup.py sdist
twine upload dist/*
