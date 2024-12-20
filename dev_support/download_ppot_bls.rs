#!/bin/bash

mkdir -p ./params
cd ./params

set -e

# Generated by GPT-4
# Function to verify and download a file based on provided hash
# $1: Filename
# $2: Expected SHA256 hash
checked_download() {
    local filename=$1
    local expected_hash=$2
    local url="https://powersoftau-transcript.s3-us-west-2.amazonaws.com/77fc8ccba8550a6c7255b82b3352bb83075fdc079a84beec8175287a6cf9b47f89f49a291025da84994753d83d9169d1b370345f367cb2dbc18b213733c5b303"

    # Check if the file already exists
    if [ -f "$filename" ]; then
        echo "File $filename exists. Verifying hash..."
        local actual_hash=$(b2sum "$filename" | awk '{print $1}')
        
        # Check if the existing file has the correct hash
        if [ "$actual_hash" == "$expected_hash" ]; then
            echo "Hash verification successful."
            return 0
        else
            echo "Hash mismatch. Expected $expected_hash, but found $actual_hash. Redownloading..."
        fi
    else
        echo "File $filename does not exist. Downloading..."
    fi

    # Download the file
    curl "$url" -o "$filename"

    # Verify the hash of the downloaded file
    actual_hash=$(b2sum "$filename" | awk '{print $1}')
    if [ "$actual_hash" != "$expected_hash" ]; then
        echo -e "\033[31mERROR: Hash verification failed after download. Expected $expected_hash, but found $actual_hash. Exiting...\033[0m"
        exit 1
    else
        echo "File downloaded and verified successfully."
    fi
}


checked_download response_21_bls12-381 77fc8ccba8550a6c7255b82b3352bb83075fdc079a84beec8175287a6cf9b47f89f49a291025da84994753d83d9169d1b370345f367cb2dbc18b213733c5b303 
