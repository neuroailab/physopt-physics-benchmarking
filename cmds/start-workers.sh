dbname=$1
num_workers=$2

for i in `seq 0 $(($num_workers - 1))`; do
	tmux new -d -s worker${i} "source ~/.virtualenvs/physopt/bin/activate && export CUDA_VISIBLE_DEVICES=${i} && ~/workspace/hyperopt/scripts/hyperopt-mongo-worker --mongo=localhost:25555/${dbname} --logfile=/mnt/fs1/workers/${dbname}-$(hostname)-worker${i}-logfile.txt"
	#sleep 1.0
done
