#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2 git netfilter-persistent
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

for port in 80 443; do
  if ! sudo iptables -C INPUT -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT 1 -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT
  fi
done
sudo netfilter-persistent save

cd "$(dirname "$0")/.."
mkdir -p data
sudo chown -R 1000:1000 data
[ -f .env ] || cp deploy/env.example .env

echo "Kurulum tamam. Docker grubunun etkin olması için oturumu kapatıp yeniden bağlan, sonra .env dosyasını doldur."
