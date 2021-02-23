# scaling_fl
This repo contains the code used to produce the results in "Scaling Federated
Learning for Fine-tuning of Large Language Models".

## Requires
Binaries [yq](https://github.com/mikefarah/yq) and `docker`.

Uses [Sacred](https://github.com/IDSIA/sacred) with a Mongo DB server for
logging.

## Usage
Before running, copy the `.env.template` file to `.env` and fill it out as you
please. To run locally, use the `run-local.sh` script. This will build and run
a container with the provided Sacred Mongo credentials set as environment
variables, allowing you to use the same container in e.g. GCP AI Platform.

### Running experiments locally

To run the agnews job with 4 sites locally, 2 on GPU 0 and 2 on GPU 1, run:

```console
> JOB_NAME=$(date +%Y%m%d_%H%M%S)
> ./run-local.sh scripts/train_federated.py \
      with \
      site_ids=[0,0,1,1] \
      num_rounds=100 \
      num_local_epochs=2 \
      iid_split=True \
      task_name="agnews" \
      train_path="/path/to/agnews/train.csv" \
      test_path="/path/to/agnews/test.csv" \
      model="distilbert" \
      job_name="${JOB_NAME}" \
      checkpoints_dir="/path/to/checkpoints/${JOB_NAME}" \
      author="${USER}"
```

If you need to store / read from a local disk set up the `$DATA_MOUNT_DIR` in
your `.env`. This will mount it to `/workspace/data`, and you'll be able to
specify paths through there. Otherwise, you can also set up GCS / S3
credentials and store / read from such paths (e.g. gs:// or s3://).

### Resuming runs
If the `JOB_NAME` matches a previous one and the checkpoints dir points to a
dictionary with checkpoints, the latest checkpoint will be loaded and the job
will continue.

### Distributed usage
We do not provide complete scripts for running across nodes. Doing so is
however relatively straightforward. You will need to make sure that the
containers run so that they are visible to each other. You then need to specify
the following [environment
variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization).

Tools like GCP AI Platform or Kubeflow can set these up for you automatically.

### Specifying clients
Internally we refer to clients as `sites`. Sites are set up by specifying a list
of (local) GPU IDs where the sites should run. If a GPU should host multiple
sites it should occur in that list multiple times.

### Specifying models
We always start from a pre-trained model and fine-tune it on either task using
FedAvg. The models are specified by providing model ids to [pre-trained
models](https://huggingface.co/transformers/pretrained_models.html) in
Huggingface `Transformers`. For convenience we have specified nicknames for the
following models which we used in our paper. You can specify these instead of
providing the full names:

| Nickname   |       Full model name |
|------------|----------------------:|
| albert     |        albert-base-v1 |
| bert       |       bert-base-cased |
| distilbert | distilbert-base-cased |

### Specifying tasks
The following tasks are available and specified in the paper:

- `agnews`
- `imdb`
- `yelp`
- `spooky_author`

Apart from specifying the task name, `agnews`, `yelp` and `spooky_author`
requires you to download the [training and test CSV
files](https://github.com/zhangxiangxiao/Crepe) and specifying paths to these.
`imdb`, however, is downloaded directly through `torchtext` and doesn't take
these parameters.

To use `spooky_author` you should download 
[the dataset](https://www.kaggle.com/c/spooky-author-identification)  and split
it into training and test subsets using the script
[scripts/spooky_author/split_data.py](./scripts/spooky_author/split_data.py).

### Debugging
Use the additional `docker-compose.debug.yaml` file to run the debug
configuration and allow you to attach a debugger to the process. When running
locally this can be done by setting the environment variable `COMPOSE_FILE`.
This will open up port `5678` and wait for a python debugger to attach before
continuing.

```console
> COMPOSE_FILE="docker-compose.yaml:docker-compose.debug.yaml" ./run-local.sh ...
```


## Citation
If you found this repo useful in your research please use the following
citation for our paper "Scaling Federated Learning for Fine-tuning of Large
Language Models":

```
@misc{hilmkil2021scaling,
      title={Scaling Federated Learning for Fine-tuning of Large Language Models}, 
      author={Agrin Hilmkil and Sebastian Callh and Matteo Barbieri and Leon René Sütfeld and Edvin Listo Zec and Olof Mogren},
      year={2021},
      eprint={2102.00875},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements
This work was part of the Vinnova project Svensk Medicinskt Språkdatalabb
(grant 2019-05156). Compute resources were provided by AI Sweden, CGIT and
Peltarion.
