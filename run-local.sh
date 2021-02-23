#!/usr/bin/env bash
set -ex

# Use the hash as a UUID
ARGHASH=`echo "$@" | md5sum - | awk '{ print $1 }'`

docker-compose build main
docker-compose -p scaling_fl-${USER}-${ARGHASH} \
    run --service-ports main $@
