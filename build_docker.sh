#!/bin/bash
# Build Docker image

if [ $# -ne 3 ]; then
  echo "ERROR USAGE: $0 <build-directory-relative-path> <dockerhub username> <cache:'' or '--no-cache'>"
  echo You need to specify a build directory and a dockerhub username
  exit 1
fi

echo working directory is $(pwd)
echo directory to build is $1

git_branch=$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
git_repo=$(basename -s .git `git config --get remote.origin.url`)
git_repo=${git_repo,,}
echo git_repo: ${git_repo}
echo git_branch: ${git_branch}

#docker build --no-cache -t ${2}/${git_repo}-${git_branch}:latest $1
docker build ${3} -t ${2}/${git_repo}-${git_branch}:latest $1
