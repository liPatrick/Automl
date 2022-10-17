### Setting up the database
We are using local postgres (to check if you have postgres installed, try and run `psql`). To install postgres, if you don't already have it, run:
```
brew install postgresql
brew services start postgresql
```
Then, you need to create the `mlmodels` database. To do this, run:
```
psql postgres
# Then, once in the shell:
CREATE DATABASE mlmodels;
```

### Running the app
```
pipenv shell
python app.py
```

### Installing a new dependency
```
pipenv install [X]
```
