#!/bin/bash

read -p "Bot token >> " token

{
  echo "TOKEN=$token"
} > ./.env