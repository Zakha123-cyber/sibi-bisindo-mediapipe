#!/usr/bin/env bash
set -euo pipefail
root=${1:?Usage: deploy.sh /absolute/deploy/root ghcr.io/owner/repo:sha-COMMIT}
image=${2:?Image required}
[[ "$root" =~ ^/[A-Za-z0-9_./-]+$ && "$root" != *'/../'* ]] || exit 2
[[ "$image" =~ ^ghcr\.io/[a-z0-9._/-]+:sha-[a-f0-9]{40}$ ]] || exit 2
[[ -f "$root/.env" ]] || { echo 'Server .env is missing'; exit 2; }
release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
previous_image=''
previous_compose=''
if [[ -f "$root/current-image" && -f "$root/current/compose.yaml" ]]; then
    previous_image=$(cat -- "$root/current-image")
    previous_compose=$(readlink -f -- "$root/current/compose.yaml")
fi
export AI_IMAGE="$image"
docker compose --env-file "$root/.env" -f "$release/compose.yaml" config --quiet
docker compose --env-file "$root/.env" -f "$release/compose.yaml" pull
if ! docker compose --env-file "$root/.env" -f "$release/compose.yaml" up -d --wait --wait-timeout 240; then
    echo 'New deployment did not become healthy.'
    if [[ -n "$previous_image" ]]; then
        echo 'Restoring the previous healthy image and Compose configuration.'
        AI_IMAGE="$previous_image" docker compose --env-file "$root/.env" -f "$previous_compose" up -d --wait --wait-timeout 240
    fi
    exit 1
fi
printf '%s\n' "$image" > "$root/current-image.tmp"
mv -- "$root/current-image.tmp" "$root/current-image"
ln -sfn -- "$release" "$root/current"
echo "Deployment healthy: $image"
