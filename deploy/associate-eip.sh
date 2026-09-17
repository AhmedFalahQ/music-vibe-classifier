#!/bin/bash
# Claim the Elastic IP that Route 53 points at, so a replaced spot instance
# takes over the same address with no DNS change.
#
# The allocation id comes from /etc/image2genre.env, written by build-ami.sh,
# so this file never needs hand-editing.
set -uo pipefail

if [ -r /etc/image2genre.env ]; then
    # shellcheck disable=SC1091
    . /etc/image2genre.env
fi

: "${EIP_ALLOCATION_ID:?EIP_ALLOCATION_ID is not set in /etc/image2genre.env}"

imds() {
    local token
    token=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60") || return 1
    curl -sf -H "X-aws-ec2-metadata-token: $token" \
        "http://169.254.169.254/latest/meta-data/$1"
}

for attempt in 1 2 3 4 5; do
    instance_id=$(imds instance-id) || { sleep 5; continue; }
    region=$(imds placement/region) || { sleep 5; continue; }

    if aws ec2 associate-address \
            --region "$region" \
            --instance-id "$instance_id" \
            --allocation-id "$EIP_ALLOCATION_ID" \
            --allow-reassociation; then
        echo "Associated $EIP_ALLOCATION_ID with $instance_id in $region"
        exit 0
    fi

    echo "attempt $attempt/5 failed, retrying in 5s" >&2
    sleep 5
done

echo "Could not associate $EIP_ALLOCATION_ID after 5 attempts" >&2
exit 1
