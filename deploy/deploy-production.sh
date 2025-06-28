#!/bin/bash
# Production deployment script for yield arbitrage system
set -euo pipefail

# Force production environment
export ENVIRONMENT=production

# Additional production safety checks
if [[ "${CONFIRM_PRODUCTION:-}" != "yes" ]]; then
    echo "WARNING: This will deploy to PRODUCTION environment!"
    echo "Set CONFIRM_PRODUCTION=yes to proceed"
    exit 1
fi

# Source the main deployment script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy.sh"