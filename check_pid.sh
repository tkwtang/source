pid=$(ps aux | grep NAND_sim | grep -v 'grep' | grep -v '.sh' | awk '{print $2, $6/1000, $12 }')
echo "$pid" | column -t

