install:
	pip3 install -e . --user

test:
	docker-compose up

ltest:
	cd maddpg
	python3 -m pytest

run:
	python3 -m maddpg.run --minio_secret=$MADDPG_MINIO_SECRET
