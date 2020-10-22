import os
import tarfile
import traceback

from minio import Minio
from minio.error import ResponseError

from maddpg.common.logger import logger


class Storage:
    def __init__(self, args=None):
        try:
            self.client = Minio(args.minio_host,
                                access_key=args.minio_key,
                                secret_key=args.minio_secret,
                                secure=False)
        except Exception as err:
            logger.error(str(err))
            traceback.print_stack()
            self.client = None
        self.bucket_name = args.minio_bucket

    def list_buckets(self):
        buckets = self.client.list_buckets()
        for bucket in buckets:
            print(bucket.name, bucket.creation_date)

    def fget_object(self, bucket_name, obj_name, save_path):
        try:
            self.client.fget_object(
                bucket_name, obj_name, save_path)
        except ResponseError as err:
            logger.error(str(err))
            traceback.print_stack()

    def list_objects(self, bucket_name, prefix, recursive=True):
        objects = self.client.list_objects(
            bucket_name, prefix=prefix, recursive=recursive)
        return objects

    def fput_obj(self, local_path, minio_path):
        try:
            self.client.fput_object(self.bucket_name, minio_path, local_path)
        except Exception as err:
            logger.error(str(err))
            traceback.print_stack()

    def make_targz(self, dst_path, src_dir):
        with tarfile.open(dst_path, "w:gz") as tar:
            tar.add(src_dir, arcname=os.path.basename(src_dir))

    def tar_and_fput(self, src_dir, dest_obj_name):
        try:
            tar_path = src_dir + ".gz"
            self.make_targz(tar_path, src_dir)
            self.fput_obj(tar_path, dest_obj_name)
            os.remove(tar_path)
        except Exception as err:
            logger.error(str(err))
            traceback.print_stack()


if __name__ == '__main__':
    from maddpg.arguments import parse_experiment_args
    args = parse_experiment_args()
    cqm = Storage(args)
    cqm.tar_and_fput("../agents", "test/agents.tar.gz")
