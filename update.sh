#!/bin/sh
#
# Deploy Ask J from this repo (~/code/askj) to /srv/askj, served by systemd as
# the www-data user. Run from the repo root:  ./update.sh
#
# Why this is more involved than a plain `cp *`:
#   - A venv is NOT relocatable: scripts in .venv/bin and pyvenv.cfg hardcode the
#     absolute path they were built at. So we never copy a venv; we build one in
#     place under /srv and keep its deps in sync with requirements.txt.
#   - .cache/ (the prebuilt embedding index) and .env (the API key) live only in
#     /srv and must survive deploys, so we exclude them from the code sync.

set -e

APP=/srv/askj

sudo systemctl stop askj

# Sync code into /srv, deleting files removed from the repo, but preserving the
# runtime-only dirs and the secret that exist only on the server.
sudo mkdir -p "$APP"
sudo rsync -a --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '.cache' \
    --exclude '.hf' \
    --exclude '.env' \
    ./ "$APP"/

# First deploy only: build the venv in place, and seed the prebuilt index so the
# service doesn't spend ~4 min embedding the corpus on its first start.
if [ ! -d "$APP/.venv" ]; then
    sudo python3 -m venv "$APP/.venv"
fi
if [ ! -d "$APP/.cache" ] && [ -d .cache ]; then
    sudo cp -r .cache "$APP"/
fi

# Keep dependencies in sync (cheap no-op when nothing changed).
sudo "$APP/.venv/bin/pip" install -r "$APP/requirements.txt"

# The service runs as www-data, so it must own everything it reads and the
# caches it writes (venv, .cache, .hf).
sudo chown -R www-data:www-data "$APP"

sudo systemctl start askj
echo "Deployed. Status:"
sudo systemctl --no-pager status askj | head -n 5
