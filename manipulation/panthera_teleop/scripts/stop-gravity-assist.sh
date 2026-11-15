#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'Stopping gravity assistance. Support both arms as assistance ends.'
exec systemctl --user stop panthera-gravity-assist.service
