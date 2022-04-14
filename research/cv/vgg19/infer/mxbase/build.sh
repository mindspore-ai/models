#!/usr/bin/bash
path_cur=$(dirname $0)

function check_env()
{
    # set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user
    if [ ! "${ASCEND_VERSION}" ]; then
        export ASCEND_VERSION=ascend-toolkit/latest
        echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
    else
        echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
    fi

    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to ./ when it was not specified by user
        export ARCH_PATTERN=./
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user"
    fi
}

function build_vgg19()
{
    cd $path_cur
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
        echo "Failed to build vgg19."
        exit ${ret}
    fi
    make install
}

check_env
build_vgg19
exit 0