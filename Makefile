install:
	pip3 install -e . --user

test:
	docker-compose up

ltest:
	cd maddpg
	python3 -m pytest

learn: clean
	python3 -m maddpg.run --save_rate=10 --warm_up 50 --batch_size 20 --env_batch_size 4

explore:
	python3 -m maddpg.run --role explorer --save_rate=10 --num_env 2 --env_batch_size 4

run: clean
	python3 -m maddpg.run --save_rate=1000 --num_env 3  --env_batch_size 50 --warm_up 1100

kill:
	ps -ef | grep maddpg.run | awk '{print $2}' | xargs kill

tb:
	tensorboard --logdir=exp/tensorboard --port=6066

clean:
	rm -rf exp
