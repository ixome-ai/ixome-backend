#!/bin/bash
# Sync local branch with Heroku and deploy

# Exit on any error
set -e

echo "Pulling remote changes from Heroku and rebasing..."
git pull heroku main --rebase

echo "Pushing changes to Heroku..."
git push heroku main

echo "Checking Heroku logs..."
heroku logs --tail --app ixome-smart-home