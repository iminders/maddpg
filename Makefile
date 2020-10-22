install:
	pip3 install -e . --user

test:
	docker-compose up

ltest:
	cd maddpg
	python3 -m pytest

learn: clean
	python3 -m maddpg.run --save_rate=1000

explore:
	python3 -m maddpg.run --role explorer --save_rate=1000

# run:
# 	nohup python3 -m maddpg.run --role explorer > /tmp/maddpg.explore.log 2>&1 &
# 	make learn

kill:
	ps -ef | grep maddpg.run | awk '{print $2}' | xargs kill

tb:
	tensorboard --logdir=exp/tensorboard --port=6066

clean:
	rm -rf exp
