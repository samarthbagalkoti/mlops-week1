ls -lh → View file sizes (important for datasets/models)

du -sh * → Check space taken by datasets/models

head -n 5 dataset.csv & tail -n 5 dataset.csv → Peek at data

grep "error" training.log → Search logs

watch -n 2 nvidia-smi → Monitor GPU usage (if GPU present)

ps -ef | grep python → See running ML scripts

kill -9 <pid> → Stop runaway training jobs
