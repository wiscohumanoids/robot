#!/usr/bin/env bash
# Stop the gravity-assist service; support both arms first.
set -euo pipefail
printf '%s\n' 'Stopping gravity assistance. Support both arms as assistance ends.'
exec systemctl --user stop panthera-gravity-assist.service
