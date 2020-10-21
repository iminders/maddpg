install:
	pip3 install -e . --user

test:
	docker-compose up

ltest:
	cd maddpg
	python3 -m pytest

learn:
	python3 -m maddpg.run

explore:
	python3 -m maddpg.run --role explorer

# run:
# 	nohup python3 -m maddpg.run --role explorer > /tmp/maddpg.explore.log 2>&1 &
# 	make learn

kill:
	ps -ef | grep maddpg.run | awk '{print $2}' | xargs kill
